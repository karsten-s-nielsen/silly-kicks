# Tracking batch performance, public seams, and DFL event derivation — design

**Status:** r6, approved by both reviewers: A = APPROVE (`…-spec-r6-A.md`) and B = APPROVE (`…-spec-r6-B.md`).
Their four r6 text notes (the cache identity rule, the float32 wording, decorator line mapping) are applied as
r6.1. The owner approved r6.1 on 2026-11-08. **r6.2 (2026-11-08):** the plan reviews found three defects inherited
from W0.3/W0.4/W3, and the owner ruled R10–R12 (section 7). The amended places carry the date. **r6.3 (2026-11-08):** owner ruling R13 (the CPU gate covers only our platforms) and the retirement of MEDIA-PC
(W0.5). **r7 (2026-11-08):** the owner added a third handoff, shape-graph positions, to the cycle. Owner rulings
R14–R16 (section 7) add W5.6 and W5.7 and amend W5.1, W5.2, NG1, NG5, G6, sections 6 and 9. **r7.1:** the r7 A/B
findings are applied (`…-spec-r7-disposition.md`). The owner amended R15: the W5.7 gate now uses a permutation null
instead of a fixed 0.80 bar. **r7.2:** NG1 and §6 name the −x exact-y-tie class. **r7.3:** the W5.7 gate uses all
20 public SkillCorner matches (owner ruling).
**r7.4 (2026-11-09):** owner ruling R17 (the empty Arm-A defect, §7) and owner ruling R18 (the W0 run
restructure: finer units, peak RSS instead of `tracemalloc`, a run budget; W0.2, §5.2, §7).
**r7.5 (2026-11-09):** the plan-r14 A/B findings (A14-PLAN-96–101, TB-PLAN-39–44) are applied to R17 and R18.
The owner confirmed the R18 budget (2 hours per leg) and ruled R19 (the W0.5 parallel limits) and R20 (the
orphaned-worker fix). **r7.6 (2026-11-09):** the plan-r15 findings (A15-PLAN-102–103, TB-PLAN-45–49) are
applied to R18 rules 3, 6, and 7 and to R20. **r7.7 (2026-11-09):** the plan-r16 finding (A16-PLAN-104 =
TB-PLAN-50) makes the R18 rule 7 takeover exclusive. **r7.8 (2026-11-09):** the plan-r17 findings
(A17-PLAN-105–106 = TB-PLAN-51–52) bound the group wait and pin the group. **r7.9 (2026-11-09):** the r18
notes (A18-PLAN-107, TB-PLAN-53), applied after A and B approved r18. **r7.10 (2026-11-09):** owner rulings R21
(sample and scale, the member slice, `profile` on the smallest match), R22 (the W0 machines: the DGX, the Mac
Mini, and HF; no local PC leg), and R23 (the staged HF launch, which replaces the job-private base tree). **r7.11 (2026-11-09):** the r20 findings (TB-PLAN-54–59, A20-PLAN-108–113) fix
the R21 slice rule, mark the R13 timing and the W0.6 CPU list as superseded, and add the R23 budget clock, CPU
stamp, CPU recovery, parity-test count, slice stamp, and upload rules. **r7.12:** the r21 B notes (TB-PLAN-60–61):
the slice digest covers the driver-side closure, and each CPU-recovery relaunch needs its own owner yes. The
r21 A notes (A21-PLAN-114–116): the same closure rule, a job limit of the budget that is left, and a relaunch
that refuses a changed campaign file. **r7.13 (2026-11-09):** owner ruling R24 (half of the largest match on
every machine). **r7.15 (2026-11-09):** the R21 slice block is 10 % per end (20 % in all), as the owner chose; 0.05 gave 10 %. **r7.14:** the r23 findings (TB-PLAN-62-63, A23-PLAN-117-120): the cut is stamped and
digested, a stamp difference is a refusal, not a gate mismatch, and the W3 peak comes from the whole match.
**r7.16 (2026-11-10):** owner rulings R25 (a W0 warm-up never fails silently), R26 (the HF W0 legs download their
data), and R27 (in-job store reads go over HTTP), from the failed base legs on `bb0c601`. **r7.17:** the r28 A/B
findings (A28-PLAN-122–125, TB-PLAN-67–72) settle the R27 read rule, bounds, retries, log tags, refusal, probe,
and docs. **r7.18:** the r29 findings (A29-PLAN-126, TB-PLAN-73–75) define the read tags, the child status record,
the bucket time field, and the probe's own-job rule. **r7.19:** TB-PLAN-76: the own-job rule uses the
ledger's `job_label`, and every reduce read of 8 MB or more must be `http`. **r7.20:** TB-PLAN-77: the own-job
rule uses a per-launch id (`SILLY_KICKS_LAUNCH_ID`), not `job_label`, and the separate reduce rule is gone
(A29-PLAN-126). **r7.21:** A32-PLAN-127: an empty launch id never matches. TB-PLAN-78: a reduced output's
writers are the `_reducers/<launch_id>.json` markers.
**r7.22 (2026-11-18):** section 7 records the owner rulings made after r7.21 (2026-11-11 to 2026-11-18), with their dates (reviewer C, TBC-DOC-07). The 2026-11-18 rulings on reviewer C's section 8 withdraw the HF leg (R22 HF part, R23, R26, R27), the W0 driver and its rulings (R10, R18-R21, R24, R25, R28), and change W2.4. The places they supersede are listed there, not edited one by one.
**Date:** 2026-11-08. **Author session:** `silly-kicks (Ragna): Optimizations` (ragnarok clone).
**Reviewers (owner-named):** A = `lakehouse (Skunk): Works`, B = `(D32):Review B-O`.
**r1 reviews:** `D:\Development\_reviews\2026-11-08-tracking-batch-dfl-events-spec-A.md`, `…-spec-B.md`.
**r1 disposition (every finding → resolution):** `D:\Development\_reviews\2026-11-08-tracking-batch-dfl-events-spec-r1-disposition.md`.
**r2 reviews:** `…-spec-r2-A.md`, `…-spec-r2-B.md`. **r2 disposition:** `…-spec-r2-disposition.md` (same folder).
**r3 reviews:** `…-spec-r3-A.md`, `…-spec-r3-B.md`. **r3 disposition:** `…-spec-r3-disposition.md` (same folder).
**r4 reviews:** `…-spec-r4-A.md`, `…-spec-r4-B.md`. **r4 disposition:** `…-spec-r4-disposition.md` (same folder).
**r5 reviews:** `…-spec-r5-A.md`, `…-spec-r5-B.md`. **r5 disposition:** `…-spec-r5-disposition.md` (same folder).
**Base:** `main` @ `af03c4b` (silly-kicks 4.139.1). Line references are at `af03c4b` unless they say otherwise. #299
moves some of them (for example `_spearman.py:193` → `:198`).
**Hard dependency:** open PR #299 (`feat/minimal-hf-driver`, read at `22e0b65`) must merge first. This design builds on
its `silly_kicks/_numba_dispatch.py` and its `scripts/_provenance.code_identity` (sections 4.0 and 4.1). If #299
changes either one before it merges, this spec is re-checked against the merged version before the plan.
**Sources:**
- `D:\Development\_handoffs\2026-11-06-sk-tracking-batch-gaps-handoff.md` (tracking gaps, items 1–6).
- `D:\Development\_handoffs\silly-kicks-dfl-event-periods\2026-11-08-silly-kicks-dfl-event-periods-handoff.md`
  (DFL event period and actor derivation).
- `D:\Development\_handoffs\silly-kicks-shape-graph-positions\2026-11-08-silly-kicks-shape-graph-positions-handoff.md`
  (frame-level per-player positions with team-relative labels), with its repro script and output in the same folder.

## 1. Intent

The v2 lakehouse scores 67,497,165 tracking frames (980 matches). Its cProfile (handoff 1, "Measured shares")
shows that silly-kicks spends most of that time in per-frame Python and pandas work, not in the numeric kernels:

- Shape graphs: about 19 ms per frame. That is about 81 min for one gradientsports match, and numba does not change it.
- Pitch control on gradientsports, with numba: 864 s. TTI is 56 s of that. Extraction takes 325 s, ball resolution
  91 s, and "velocity" 182 s. "Velocity" is `smooth_frames` + `derive_velocities` together (reviewer A, from the v2
  profiler script).

These are the handoff's figures. Its gradientsports frames then held duplicate player rows (lakehouse spec 7.3a). The
C1 record measured the same calls on clean frames, before and after this cycle, in
`docs/research/tracking-compute-perf/README.md`.

Separately, a consumer that parses DFL event XML itself has no public way to get each event's period,
period-relative time, and acting player and team. The v2 lakehouse located events by `calculated_frame` (which
only shots carry) and got 18–27 SPADL actions per idsse match from about 1,500 events, with no error.

**The owner's bar is "gold standard, best practice". "Done" means both of these:**

1. The lakehouse's next full run is fast and correct.
2. The lakehouse imports no private silly-kicks module for any of this work.

The work is general library functionality. It must not be shaped to one consumer (lakehouse v1 or v2).

## 2. Goals, non-goals, anti-goals

### Goals

- G1. Public, documented, typed entry points for every capability the lakehouse reaches today through a private
  module: the numba backend query, chunked and indexed pitch control, multi-frame shape graphs and positions, line
  breaking (already public; docs fixed), and DFL event period, time, and actor derivation.
- G2. Remove the per-frame pandas work from:
  - the batched pitch-control path;
  - the shape-graph path;
  - the line-breaking path;
  - `smooth_frames` and `derive_velocities`.
- G3. Every speed change is **bit-identical** to the current output (section 5). It is proven on synthetic
  production-shape fixtures in CI, on the real corpus once on x86, and once on arm64, with recorded results.
- G4. A measured before/after record (wall-clock, per step, per provider) in `docs/research/`.
- G5. One implementation per concept:
  - The DFL XML parser and the new DataFrame-level functions share one rule for period and one rule for actor.
  - The batched and per-frame pitch-control paths share one numeric core.
  - Every caller of the private batch kernel (`_cover_shadows`) moves to the shared index.
- G6. Correctness fixes the owner approved (section 7):
  - the duplicate/multi-game `action_id` defect in `detect_line_breaking` and the shape-graph action paths (R1, R6);
  - the ADR-019 string id keys in the code this cycle rewrites (R2);
  - the multi-game defect at its root, `link_actions_to_frames`, and in every package consumer of its pointers
    (R5, W9);
  - the position labels of `infer_positions`, corrected in place to the team's own frame (R14, W5.6), with a gate
    on real data (R15, W5.7).
- G7. A multi-game invariant, gated for every public function that links actions to frames:
  `f(game_a ∪ game_b)` equals `concat(f(game_a), f(game_b))` in values.

### Non-goals

- NG1. No change to any metric value, default, or trained artifact. No retrain. Section 7 lists the only
  approved exceptions (R1, R2, R5, R6, R14). R1, R2, R5, and R6 change values only for the inputs they name:
  multi-game inputs, duplicate-key inputs, and id-dtype-mismatch inputs. Single-game, unique-key, dtype-consistent
  inputs stay bit-identical everywhere, except for R14. R14 changes the labels of `infer_positions` (and so of the new
  `compute_shape_graphs`) by the exact relations in W5.6: the lateral labels of every +x team, the D-tree vertical
  case of a −x team, the lateral fallback labels of a −x team with n ∉ {5, 10} players, and the lateral labels of
  two −x middle players with exactly equal y (the index tie-break; plan round 11, TB-PLAN-33). No `add_*` output and no `*_xfns` output changes, because
  none of them calls `infer_positions`.
- NG2. No new metric family and no new `add_*` enricher.
- NG3. No GPU work. No multiprocessing inside the library. Callers own parallelism (ADR-128 single-threaded units).
- NG4. No change to the DFL bronze that `parse_dfl_events` emits: schema, values, row set, row order, or dtypes
  (a versioned cross-repo contract, ADR-031 N1).
- NG5. No cooperative stop callable on `detect_line_breaking` (owner ruling) or on `compute_shape_graphs` (R16).
  Chunked calls give callers stop points.
- NG6. The numba unification itself (one dispatch point, numba core, fail-loud) is PR #299's work. This cycle does
  not redo it.

### Anti-goals (failure even if it "works")

- AG1. A speed-up that changes any output bit without an explicit owner ruling.
- AG2. A second implementation of a rule inside the package: period derivation, actor order, spearman combine,
  shape-graph tie-breaking, or the Kalman recursion. Frozen pre-change copies under `tests/` are test oracles, not
  implementations. Nothing in `silly_kicks/` imports them (section 5.4).
- AG3. A public API that names consumer columns (`kick_off_game_section`, `_IDSSE_PLAYER_COLS`) or consumer tables.
- AG4. A silent fallback or a silent drop. Every skipped, null, or unresolved row is counted in a report or warning.
- AG5. A wall-clock assertion in CI (flaky). CI guards cost with structural guards (call counts, ADR-073).

## 3. Constraints (from AGENTS.md and the repo)

- Hexagonal purity: pure functions, pandas in and pandas out, no I/O, no global state mutation. `add_*` are
  input-mutation-gated (`PURITY_ENTRIES`).
- Ids: `silly_kicks.id_compat` only. No raw `==`/`!=` on ids across sources, no unaligned id merges, no `astype(str)`
  or `str()` on an id used as a dict key or join token (ADR-019/043). A comparison of a column with values taken from
  that same column (`tid in frames["team_id"].unique()`) is not a cross-source comparison. It is left exactly as
  it is (section 7, R2).
- Orientation: direction comes from the frames' labels, never from team identity. Unresolved is a `<NA>` value
  (ADR-051/055).
- Frames: coordinates are float32 storage and float64 compute. `team_id` is `category` (ADR-106).
- No rescan-in-loop: use `silly_kicks._frame_index.group_rows` (ADR-068). Item-looping primitives carry
  `assert_subquadratic_growth` with a scoped counter registered in `SCALE_GUARDED` (ADR-073).
- Frame-aware `*_xfns` resolve frames by position (ADR-020).
- Private modules can have downstream consumers. `docs/PRIVATE_CONSUMERS.md` lists these rows that this cycle touches:
  - `_frame_index.py` (`group_rows`, `RowGroups`; `:49`);
  - `tracking/preprocess/_butterworth.py` (`:47`);
  - `tracking/_cover_shadows.py` (`lane_control`, `LaneControlResult`, `CoverShadowParams`; `:45`).

  None of the listed symbols may change signature or behaviour. A new consumer of `_frame_index` gets a row added.
- Hyrum's law: return types, column names, dtypes, warning categories and texts, and the element order of returned
  containers (for example `ShapeGraph.faces`) are API. A new warning category is a new observable surface. It is
  documented and listed in the CHANGELOG.
- Process:
  - Work on one feature branch from `main`, created after #299 merges.
  - Make one commit for the whole cycle, only after `/final-review` and an explicit owner yes.
  - Run an A/B review of the spec, the plan, and the implementation.
  - Assign the version number, ADR number, and `PR-Snnn` key at commit-prep. ADR-130/131 are already claimed on
    open branches.

## 4. Workstreams

Each workstream states what the code does today, the design, and its acceptance.

### 4.0 W0 — Measure first, measure after

The handoff's numbers come from the lakehouse's profiler on an arm64 Mac. silly-kicks needs its own baseline before
any change.

- W0.1 A corpus driver `scripts/measure_tracking_compute.py` on the `scripts/_driver.py` seam (ADR-052: resume before
  load, `for_each`, provenance). It registers in `ARTIFACT_DRIVERS` and gets the registry gate and a driver unit test,
  like every other artifact driver (ADR-056).
- W0.2 For the largest match of each tracking provider in the corpus (gradientsports, idsse, skillcorner, plus
  metrica and pff where present), it records wall-clock and cProfile cumulative time for each of these steps:
  - `smooth_frames`, and `derive_velocities` separately;
  - `compute_pitch_control_batch`, split into TTI, extraction, ball, influence and combine, and surface assembly;
  - per-frame `compute_pitch_control`, the path v2 uses today, for the handback comparison;
  - `compute_shape_graph` and `infer_positions`;
  - `detect_line_breaking`.

  The driver also runs the corpus-scale parity passes of section 5.2.

  **Amended 2026-11-09 (owner ruling R18).** The W0 run has finer units, measures memory as peak RSS, and has a
  run budget. Section 7, R18, states the rules. The step list, the backends, the members, the matches, and the
  machines do not change.
- W0.3 It runs on the baseline. **Amended 2026-11-08 (owner ruling R10; plan reviews A1-PLAN-45 and TB-PLAN-01).**
  The driver does not exist at `BASE_SHA`, and the cycle has one commit, so "a clean tree at the branch base" cannot
  run it. The base rule is now:
  - The **package** identity must equal `BASE_SHA`'s. The scoped hash of `silly_kicks/` (less `_version.py`, R11),
    `pyproject.toml`, and `uv.lock` in the working tree must equal the same hash built from the `BASE_SHA` commit.
    If it differs, the base phase refuses.
  - The new, uncommitted `scripts/` and `tests/` files (the driver, its fixtures, the W9 population) are allowed.
  - The base phase stamps `BASE_SHA`, the package hash, and the `scripts/` hash.
  - The base run comes after the W9 population exists and before the first package edit.
- W0.4 It runs again on the final code before the cycle commit. One commit per cycle means that tree is still
  uncommitted, so the after-run uses `allow_dirty=True` and stamps `run_tree_dirty=True`.
  - `code_identity` (`scripts/_provenance.py:264`) and `git_tree_hash` (`:54-78`) hash the whole tree, untracked
    files included. The memo and the docs change the tree after the run, so a whole-tree identity cannot match the
    commit (reviewer B).
  - W0 therefore stamps a **scoped code identity**: a content hash over exactly `silly_kicks/`, `scripts/`,
    `pyproject.toml`, and `uv.lock`, tracked and untracked files alike. **Amended 2026-11-08 (owner ruling R11):**
    `silly_kicks/_version.py` is left out. It holds only the version string, so the commit-prep version bump does
    not change the identity.
  - Step 8 of section 9 ends all code edits. Docs, the ADR, and the memo come after it and are outside the scope.
    Every in-scope edit that the plan knows of (the W3 measured peak, the gated-candidate defaults, the public
    re-exports) lands before the freeze.
  - At commit-prep, the scoped hash of the staged cycle commit must equal the after-run's stamp. A mismatch means
    the after-run is redone. A git failure never compares equal (fail-closed).
  - The scoped hash is a small addition to `scripts/_provenance.py`, with a unit test. The memo records it, the
    dirty flag, and the base commit. This is the ADR-037 dirty-run route: explicit and recorded, not silent.
- W0.5 **Amended 2026-11-08:** MEDIA-PC is permanently gone. The x86 runs happen on the local PC, one match at a time.
  Before each heavy run, the builder states the expected peak memory to the owner, because the local PC crashes under
  memory pressure. The arm64 runs go to the owner's Mac Mini (owner ruling R4). Data comes through pining.
  **Amended 2026-11-09 (owner ruling R19):** the R18 `time` stage still runs one unit at a time. The other R18
  stages may run in parallel within the R19 limits (section 7).
  **Amended 2026-11-09 (owner ruling R22):** the W0 legs run on the DGX and the Mac Mini (arm64), and on HF
  `cpu-upgrade` (AMD x86), staged per R23. The local PC runs no W0 leg.
- W0.6 Results land in `docs/research/tracking-compute-perf/`, with the machine, CPU class (AMD x86 AVX2, arm64; R22
  NEON), numpy and numba versions, and the commit or content identity.

**Acceptance:** a committed before/after memo with x86 and arm64 legs. The driver test is green. No speed target is
asserted in CI (AG5). The owner sees the measured gain at commit-prep.

### 4.1 W1 — Public numba backend query (handoff item 1)

**After #299 (read at `22e0b65`):**
- `silly_kicks/_numba_dispatch.py` is the one dispatch point for six families: `GHOST`, `DAS`, `COORDINATION`,
  `PITCH_CONTROL`, `BALL_CARRIER`, `XTGK`.
- numba is a core dependency. `use_numba(family)` raises `NumbaUnavailableError` when numba cannot be imported,
  so the silent fallback is gone.
- `SILLY_KICKS_<FAMILY>_FORCE_NUMPY=1` selects the numpy reference for one family.
- The module is private, so a consumer cannot ask which backend a family uses without importing it.
- #299 also removes `_HAS_NUMBA`, which a lakehouse guard reads today. That goes in the handback (section 6).

**Design:**
- A public module `silly_kicks.backends`:
  - `active_backend(family: str) -> Literal["numba", "numpy"]`. It has the same semantics as `use_numba`: it raises
    `NumbaUnavailableError` when numba is missing and the family is not forced to numpy, and `ValueError` for an
    unknown family.
  - `backend_report() -> dict[str, str]`. It is a diagnostic, so it **never raises**. It maps each family to
    `"numba"`, `"numpy"` (forced), or `"unavailable"` (numba missing), plus the key `"numba_version"` (a version
    string or `"unavailable"`). It is for logging and environment guards.
  - Re-exports: `FAMILIES`, `NumbaUnavailableError`.
- `_numba_dispatch` stays the implementation. `silly_kicks.backends` is a thin public façade, so #299's call sites
  do not move.
- Any new numba kernel this cycle adds (W5.1, W7) registers its **own new family** in `FAMILIES`, with its own
  `SILLY_KICKS_<FAMILY>_FORCE_NUMPY` switch and its own parity test. The family count is whatever results.

**Acceptance:** unit tests for each family under each switch state and under a simulated missing numba. These cover
the raise from `active_backend` and the non-raising `"unavailable"` from `backend_report`. Doctests on the public
functions. `docs/context/conventions-core.md` gains a "backends" paragraph.

### 4.2 W2 — Batched pitch control without per-frame pandas (handoff item 2)

**Today:**
- `_batch_from_groups` (`pitch_control/_dispatch.py:212`) dedups requests on
  `(tuple(frame_key), str(attacking_team_id), bool(decompose))` (`:234`). That is a `str()` id key; see R2.
- It calls `groups.get` per distinct **request** (`:245`), not per distinct frame. A frame requested for both teams,
  or with and without `decompose`, is sliced, extracted, and TTI-scored more than once.
- `_resolve_ball_position` does a boolean mask on the slice per request (`:281`).
- `compute_spearman_batch` (`_spearman_batch.py:61`) calls `_extract_frame_players` per request: a `.copy()`, a
  `dropna`, and four `to_numpy` (`_spearman.py:193-213`).
- TTI is one call over the concatenated players of all requests (`:66-70`).
- `ids_match` and `_spearman_combine` run per request (`:76-99`).
- The private `compute_spearman_batch` and `_resolve_ball_position` have a second caller, `_cover_shadows.py:941-954`.
  It scores counterfactual frame **slices** that are not in any frame-keyed grouping (the ADR-043 landmine).

**Design:**
- W2.1 **Columnar frame index.** One internal `_SpearmanFrameIndex` holds:
  - contiguous float64 `x`, `y`, `vx`, `vy`, upcast from float32 storage once at the boundary (ADR-106);
  - bool `is_ball` and `is_goalkeeper`;
  - the `player_id` and `team_id` arrays;
  - per-frame row offsets in original frame-row order.

  It can be built from a `group_rows` grouping (keyed frames) or from an explicit list of frame slices
  (`_cover_shadows`). The per-frame filter "not ball, and x and y not NaN, in frame order" is one vectorised mask
  plus offsets. It must equal `_extract_frame_players` element for element, including the `vel` NaN→0.0 rule and
  the "no `vx` column → zeros" rule.
- W2.2 **Per-frame, not per-request.** Extraction, ball resolution, and TTI run **once per distinct frame slice**.
  TTI does not depend on the attacking team. The combine runs per distinct request on that slice's TTI rows. This is
  bit-identical, because the same inputs reach the same `compute_tti` call per player row.
  - A keyed index (built from `group_rows`) dedups by frame key.
  - A slice-built index (`_cover_shadows`) **never dedups by frame id**. It treats every slice as distinct, because
    `_cover_shadows` batches a frame and its keeper-removed counterfactual, which have the same frame key and
    different content (`_cover_shadows.py:908-954`, ADR-043). The fixture includes two slices with one frame key.
- W2.3 **Ball** from the index: the first ball row per frame, and NaN → `None`. This equals
  `_resolve_ball_position(fs, None)`.
- W2.4 **`is_attacking`** comes from `id_compat.ids_match`, called **per request on that request's frame-slice
  view**, as today (`_spearman_batch.py:83`).
  - **Changed by owner ruling 2026-11-18 (section 7, C3; reviewer C TBC-PERF-03).** One `ids_match` runs per
    (index part, team) over that team's requested rows when no slice can change its path:
    `id_compat._ids_match_path_is_content_free` (a non-object column, or an object column of strings only).
    An object column with non-string ids keeps the per-request call below. The decomposed `player_team_ids`
    use one `to_numpy` per part when no slice can change its dtype. Surfaces stay bit-identical, and the
    mixed-content fixture still passes. The text below is the rule for the per-request case.
  - `ids_match` picks its raw or canonical path from the content it sees (`id_compat.py:188-217`, `:425-435`).
    So one call over the whole batch is not equivalent, and it is not allowed (reviewer B).
  - A view slice costs O(slice rows). The total work is the sum of the requests' slice lengths, which is the same as
    today and is not a rescan.
  - A structural guard asserts that the total elements passed to `ids_match` equal that sum.
  - A mixed-content fixture puts frames whose `team_id` content takes the raw path in the same batch as frames that
    take the canonical path.
- W2.5 **Batched combine.**
  - These steps run over the concatenated rows of the chunk:
    - the opponent-minimum TTI per (frame, side): `min` is exact and order-free;
    - the influence, through one named new kernel `_influence_rows(tti_rows, opp_min_rows, sigma)`. It is
      elementwise, and each row carries its own frame's opponent minimum;
    - the ball-travel `too_slow` mask and the GK weighting.

    `_influence_rows` has a numpy reference and a numba kernel in the `PITCH_CONTROL` family. Both are
    parity-gated against today's `_compute_influence` per frame (section 5).
  - **The sums stay sequential.** The order-dependent step is the influence `.sum(axis=0)`. Its bit-identity rests
    on summing rows sequentially in the players' row order. Reviewer A measured this: a C-order axis-0 sum is
    sequential at any row count (0/2,000 differ). `np.add.reduceat` and transposed sums are not (2,000/2,000 differ).
    The 128-element threshold in the `_spearman_combine` docstring is not the invariant. The docstring is corrected.
    W2.5 therefore computes each frame's sums with a C-order axis-0 `.sum` over that frame's contiguous row
    slice. `reduceat` is not allowed. An empty side (no attackers or no defenders) takes today's
    `np.zeros((0, n_targets))` path. The fixtures include such frames, because `reduceat` on an empty segment
    would silently return the next row.
  - **Decompose** is batched the same way. `per_player_influence` is filled by one scatter per frame from the
    batched influence rows, not by a per-row Python loop.
  - Python then runs only to assemble one `PitchControlSurface` per request. That loop is unavoidable, because each
    request returns its own object.
- W2.6 **Single source.**
  - `compute_spearman` (per frame) builds a one-frame index and calls the same core.
  - `compute_spearman_batch` keeps its signature and becomes a thin adapter over the slice-built index, so
    `_cover_shadows` keeps working unchanged in behaviour.
  - `_cover_shadows` moves to the index adapter.
  - `PitchControlCache.warm` uses the same index (ADR-105 Task 5).
  - `_spearman_combine` stays as the per-frame reference: in `tests/` as a frozen oracle, and in the package only if
    a per-frame caller still needs it. The plan decides which, under AG2.
- W2.7 The velocity-required `ValueError` and the empty-frame uniform-0.5 surface stay exactly as they are.

**Acceptance:**
- These are bit-identical (section 5.1) against the pre-change per-request loop on both backends:
  - the existing batch parity test (`test_spearman_batch.py`);
  - a new production-shape fixture (section 5.2).

  The fixtures cover:
  - with and without `decompose`, and with and without a ball;
  - an empty frame, a frame with no attackers, and a frame with no defenders;
  - both-team requests on one frame, and duplicate requests;
  - mixed id dtypes.
- `_cover_shadows` lane-control outputs are bit-identical before and after on its existing tests and on a seeded
  counterfactual fixture.
- Structural guards per chunk:
  - `compute_tti` runs once;
  - `_influence_rows` runs a constant number of times, independent of the frame count (decompose included);
  - `_extract_frame_players` runs 0 times on the batch path;
  - extraction runs once per distinct frame, however many requests name that frame.

### 4.3 W3 — Memory-bounded chunking and a public index (handoff item 3)

**Today:** `compute_pitch_control_batch(frames, requests)` groups the whole `frames` on every call
(`_dispatch.py:207-209`). It materialises one TTI intermediate for every distinct request in the batch
(`_spearman_batch.py:11-12`). The pre-grouped seam `_batch_from_groups` is private.

**Design:**
- Add a keyword-only `max_frames_per_chunk: int | None = 2048` to `compute_pitch_control_batch` (owner ruling F2).
  The function groups once and builds the index once. It then runs the kernel over the distinct **frames** that the
  requests name, in chunks of at most that many frames. A chunk holds every request on its frames. `None` means one
  chunk. Values are bit-identical at any chunk size, because each surface depends only on its own frame.
- Memory statement, for the docstring:
  - TTI is per distinct frame (W2.2): at most 2,048 × 22 players × 1,600 targets (default 50 × 32 grid) × 8 bytes
    ≈ 0.58 GB.
  - The influence, mask, and per-player intermediates add a small multiple of that.
  - On the numpy backend, `compute_tti` broadcasts (n_players, n_targets, 2) displacement arrays. Its peak is
    several times the TTI size (reviewer B).

  The docstring states the measured peak per backend from W0, and tells a memory-tight caller to lower the chunk size.
  (Clarified 2026-11-08: the peak comes from a W0 measurement on the candidate code just before the freeze. The
  docstring states it as a rounded-up bound, and the after-run checks that the measured peak does not exceed it.)
- A public frozen `PitchControlFrameIndex` and `build_pitch_control_index(frames) -> PitchControlFrameIndex`, plus a
  separate public function `compute_pitch_control_batch_from_index(index, requests, *, method, params,
  max_frames_per_chunk)`. A caller that scores one match in several request waves pays the grouping once, with no
  private import. The first parameter of `compute_pitch_control_batch` is **not** overloaded (a Hyrum surface).
- Non-spearman methods (`voronoi`, `fernandez_bornn`) work through the same two functions with unchanged values. The
  index keeps the `RowGroups`, and those methods take their frame slices from it exactly as today. Batching their
  kernels is out of scope.
- Validation: `max_frames_per_chunk` must be a positive `int` or `None`, else `ValueError`.

**Acceptance:**
- Bit-identity across chunk sizes 1, 7, 2048, and `None`, for all three methods.
- Equality between the frames path and the index path.
- A structural guard: `group_rows` runs once per call, however many chunks there are.
- The docstring's memory statement matches the W0 measurement.

### 4.4 W4 — Backend parity record (handoff item 5)

**Today:** `tests/tracking/pitch_control/test_numba_parity.py` asserts numba vs numpy on small seeded fixtures:
`rtol=1e-12` for TTI and influence, `rtol=1e-10` for Fernández-Bornn. Handoff item 5 is partly wrong, because that
test exists. These are still missing:
- a production-scale case;
- a stated maximum absolute difference;
- the at-target branch: numpy `np.where(distance < 1e-10, …)` vs the numba scalar branch (`_spearman.py:63-83` vs
  `_numba_kernels.py:72-83`);
- a document.

**Design:**
- For every numba kernel this cycle touches or adds (`tti_numba`, `influence_numba`, `_influence_rows`, and any W5.1
  or W7 kernel), extend the parity tests with:
  - a production-shape case (section 5.2);
  - an explicit at-target case (`distance == 0` and `0 < distance < 1e-10`);
  - an assertion of the measured `max_abs_diff` against a stated bound.

  Ball-carrier kernels are not touched by this cycle and are not in W4. Their parity test,
  `test_ball_carrier_numba_parity.py`, stays as it is.
- The corpus-scale numbers (max abs diff per kernel, per provider, per CPU class) come from the W0 driver. They land
  in the W0 memo, and `docs/context/tracking-features.md` points to them.
- The tolerance stays what the tests assert today, unless the measurement shows a larger real difference. Then the
  builder stops and brings the measurement to the owner (section 5.3).

**Acceptance:** tests green on CI, and the memo states the bound and the measured maximum per kernel and CPU class.

### 4.5 W5 — Shape graphs: public multi-frame API and a faster core (handoff item 4)

**Today:**
- `compute_shape_graph`, `infer_positions`, `ShapeGraph`, and `PositionLabel` are **already public** as
  `silly_kicks.tracking.*` (`tracking/__init__.py:416`, `__all__`). Handoff item 4's "no public seam" comes from
  docstrings that teach `from silly_kicks.tracking._shape_graph import ...` (`_shape_graph.py:320`, `:903`).
- There is no multi-frame entry point.
- The per-frame cost is in pure Python:
  - `_angle_at_vertex` does numpy scalar calls (`np.dot`, `np.linalg.norm`, `np.arccos`) per opposite vertex.
  - `_compute_edge_stability_from_faces` and `_merge_faces_for_edge` scan every face for every edge (O(E·F)).
  - `_update_stabilities_for_merged_face` scans every active edge.
  - `infer_positions` builds up to two more shape graphs through `_decompose_middle` (`:599`), at most one per axis
    and only when that axis has three or more middle players.
- `add_shape_graph` (`features.py:5491+`) and `shape_graph_xfns` (`:5639+`) each re-scan the whole `frames` per team.
  Their `frames["team_id"] == tid` uses a `tid` from the same column's `unique()`, so it is not a defect (R2). Each
  also re-keys `linked["game_id"].astype(str)` (`:5589`, `:5749`), which is the ADR-019 defect R2 fixes.
- **Label defects (handoff 3).** The lakehouse scored the labels against SkillCorner `player_role` on three
  open-data matches (handoff table). Three defects are in `infer_positions`:
  - **D-lat. The lateral ladder is inverted for a team that attacks +x.** `_assign_levels_horizontal` gives "R" to
    high y (`:796-801`). In SPADL (ADR-004, origin bottom-left), high y is the left side of a team that attacks +x.
    The measured home lateral agreement is 0/7, 2/11, and 1/12. For a team that attacks −x, the pitch-absolute label
    is correct only by accident (8/9, 9/11, 8/11). ADR-045 D5 settled the pitch-absolute convention by default,
    because no consumer existed. The lakehouse v2 is now that consumer.
  - **D-eqf.** The equal-frequency fallback has the same inversion: `_equal_frequency_assign` sorts ascending y onto
    `_HORIZONTAL_LEVELS = ("L", …, "R")` (`:786`, `:838-855`).
  - **D-tree. The pitch-thirds branch flips x twice for a team that attacks −x.** `infer_positions` passes `x = -x`
    as `values` (`:927-928`), and the tree branch of `_decompose_middle` negates the mean again (`:674-676`). So a
    deep middle group of that team (high original x) gets `label_above` ("AM") instead of "DM". The branch runs only
    when the middle sub-graph is a tree. It is not proven to explain the away vertical gap (handoff, "Not explained
    yet").
  - A non-finite or zero `attacking_direction` is read as +1 today (`attacking_direction < 0` is false). That is a
    silent guess (ADR-051, AG4).

**Design:**
- W5.1 **A faster core, bit-identical (section 5).** The builder chooses the technique under the gate.
  - These are allowed:
    - an edge → incident-face index maintained across merges;
    - caching the opposite-angle values per (edge, face);
    - replacing numpy scalar calls with `math`-module scalars, **but only for correctly-rounded operations**
      (`+ − × ÷`, `sqrt`).
  - `acos`, `atan2`, and `degrees` are not correctly rounded, and numpy can dispatch them to SIMD code per CPU class.
    A replacement for them ships only if the gate passes on **every platform we run** (§5.2, R13): AMD x86 AVX2 (HF cpu-upgrade) and arm64 (the Mac Mini and the DGX) (R22: the local PC left the rule).
    Otherwise those calls stay numpy calls with the same inputs. "Not testable" counts as "not proven" (reviewer A).
  - A numba kernel is allowed under the same rule, as its own family (W1), with the Python version kept as the
    reference.
  - The core must keep these exactly:
    - the set and dict iteration order that drives tie-breaking (`min_edges[0]` is `e0`);
    - the 1e-10 tie tolerance;
    - the order of `faces` in the returned `ShapeGraph`;
    - the edge sort;
    - the `QhullError` warning.
- W5.2 **A public multi-frame entry point** in `silly_kicks.tracking`:
  `compute_shape_graphs(frames, *, stability_threshold=45.0, with_positions=True, retain_graphs=False) -> ShapeGraphBatch`.
  - **Selection** is the one `add_shape_graph` uses today: per (game_id, period_id, frame_id, team_id), outfield rows
    only (`not is_ball`, `not is_goalkeeper`, `x`/`y` not NaN), in frame-row order. Teams come from the frames' own
    `team_id` values, as in `add_shape_graph`. No id is merged across dtypes.
  - **Direction** for `infer_positions` comes from the frames' `team_attacking_direction` label (ADR-051; domain
    `{"ltr", "rtl"}`, `tracking/schema.py:168`): `"ltr"` → `+1.0` and `"rtl"` → `-1.0`. Any other value gives `<NA>`
    labels and `shape_graph_source="direction_unresolved"`, never a guess. The graph does not depend on direction, so
    the summary metrics are still emitted.
    - **Amended r7 (handoff 3; TB-SPEC-57).** All selected rows of one team-frame (the outfield rows that get labels;
      the ball and goalkeeper rows are not read, TB-SPEC-60) must carry the same direction value, null included. Any mix (two domain values, or a domain value with a null or another value) is a data defect.
      The function raises `ValueError` and names the first such (game_id, period_id, frame_id, team_id). This
      follows the `add_das` precedent for a partly populated direction (`_validate_per_frame_attacking_direction`,
      `features.py:2927`). A team-frame whose rows are all null or all one non-domain value is unresolved. Different
      values across frames or periods are normal.
    - The handoff asked for no rows on an unresolved team-frame. The `<NA>` rows plus the source give the same
      information and keep one row per selected player (accepted by reviewer B).
    - **`players` also carries `shape_graph_source`** (TB-SPEC-55), the team-frame's value repeated on each of its
      rows. So a consumer selects `players[players["shape_graph_source"] == "ok"]` with no id join (ADR-019). With
      `with_positions=False` the source is still set, and the labels are `<NA>` on every row. The docstring and the
      handback name this filter.
    - **All-or-nothing:** a team-frame's labels are all set or all `<NA>`. The core asserts that `infer_positions`
      returned one label per selected player when the source is `"ok"`. A mismatch is an internal error, not a
      partial row set.
  - **`ShapeGraphBatch`** (frozen) has two DataFrames:
    - `frames_summary`: one row per (game_id, period_id, frame_id, team_id), with `n_players`, `n_edges`, `density`,
      `mean_stability`, and `shape_graph_source` ∈ `{"ok", "too_few_players", "degenerate", "direction_unresolved"}`.
      `"degenerate"` covers a Delaunay `QhullError` and a graph with no edges. The domain is a public constant
      `SHAPE_GRAPH_SOURCE_VALUES`, following the `DAS_SOURCE_VALUES` precedent. It is **not** in
      `TRACKING_CATEGORICAL_DOMAINS`, whose keys must be frame columns (`tests/test_tracking_schema.py:179`;
      reviewer A).
    - `players`: one row per (game_id, period_id, frame_id, team_id, player_id) for **every** selected outfield
      row, with `shape_graph_source`, `vertical`, `horizontal`, and `position_label`. The labels are `<NA>` when the
      frame's source is not `"ok"` or `with_positions=False`.

    **Row order** is the order of each team-frame's first appearance in `frames`, which is the `group_rows` order.
    Players are in frame-row order inside a team-frame. Nothing is sorted on id values, so a mixed-dtype `game_id`
    cannot raise (reviewer B). The docstring states the order.
  - **Memory:**
    - `retain_graphs=False` (default): no `ShapeGraph` objects are kept. `ShapeGraphBatch.graph(game_id, period_id,
      frame_id, team_id)` recomputes the graph from the retained positions. It is deterministic and bit-identical to
      the batch's own result.
    - `retain_graphs=True`: every graph is kept. The docstring gives the cost per team-frame, measured in W0, and
      warns that about 508k team-frames per large match is several GB.
  - **Looping:** it loops over team-frames once from a single grouping (ADR-068). The Delaunay stays per frame.
  - **Chunked calls (R16).** Team-frames are independent. So for chunks that are contiguous row slices of `frames`
    and split on whole frames, that is on (game_id, period_id, frame_id) boundaries, the concatenation of the
    per-chunk results equals one call on all the frames: both DataFrames, row order included. Both conditions are
    needed, because the row order is the first-appearance order (A7-SPEC-89). The docstring shows the chunked pattern for a caller with a deadline.
    The function has no stop callable (NG5).
- W5.3 **Single source.** `add_shape_graph` and `shape_graph_xfns` compute through the same per-team-frame core as
  W5.2.
  - The direction is read and checked (resolution, the mixed-value `ValueError`) only on the label path, that is
    `compute_shape_graphs(with_positions=True)`. The graph core never reads it. So `add_shape_graph` and
    `shape_graph_xfns` do not newly raise on a mixed-direction input, and NG1 holds (A7-SPEC-90). With
    `with_positions=False`, the source is never `"direction_unresolved"`.
  - **R6:** their action paths (`add_shape_graph`'s linked merge and `_shape_graph_at_actions`) merge on
    `action_id` alone (`features.py:5576-5580`). For a duplicate id, `aid_to_idx.loc` returns a Series (`:5591`).
    Both move to the W9 positional join on `action_row` and the game-keyed frame lookup, so each row gets its own
    result. Gamestate slots are covered.
  - Their outputs stay bit-identical on single-game inputs with unique ids and consistent `game_id` dtypes. They
    change only for the R2 (re-key), R5, and R6 input classes, each with a detection test first.
- W5.4 **Docs.** Fix every docstring example in the F3 scope to import public symbols from their public path
  (`_shape_graph.py:320`, `:903`; `_line_breaking.py:47`, `:97`; others the test finds). Add a test that fails if a
  docstring example in the F3 scope imports a private module for a symbol with a public path.
- W5.5 `feature_glossary` entries for the new columns. The `NOTICE` entry for Sotudeh (2026) already exists.
  The `vertical`, `horizontal`, and `position_label` entries state the W5.6 convention.
- W5.6 **Team-relative position labels (handoff 3; owner ruling R14).** `infer_positions` is corrected in place.
  Its signature does not change.
  - **Convention.** Every label is in the team's own frame, with SPADL handedness (ADR-004).
    - Vertical: "B" is the team's own goal end, as today.
    - Lateral: for a team that attacks +x, "L" is high y. For a team that attacks −x, "L" is low y. So "L" is the
      left of a player who faces the goal that the team attacks.
  - **One attack-frame map.** `attacking_direction > 0` means the team attacks +x, and `< 0` means −x. A non-finite
    or zero value raises `ValueError` (it is never a guess). The positions go first through the existing ADR-051
    point reflection, `to_goal_relative_x_array` and `to_goal_relative_y_array` (`tracking/_geometry.py:204`, `:218`): the
    identity for +x, and `(105 − x, 68 − y)` for −x. Every label input comes from these attack-frame positions:
    - the player values;
    - the face centroids (the caller's `shape_graph` supplies only the face vertex sets and the edge count);
    - the bridging-edge midpoints and the middle sub-graph of Option C, which is built from the attack-frame
      positions of the middle players;
    - the middle-group mean for the pitch-thirds rule.

    A face centroid sums its vertices in ascending vertex-index order, so it does not depend on the face's vertex
    order.
  - **Ladders in the attack frame.** Vertical, ascending x: `B, DM, M, AM, F` (as today). Lateral, ascending y:
    `R, RC, C, LC, L`, because in the attack frame high y is the left. The face-centre path, the Option D percentile
    path, the middle decomposition, the pitch-thirds rule, and the equal-frequency fallback all use these ladders.
  - The `flip` arguments are removed from the private helpers, because the attack frame needs no flip. This fixes
    D-tree: the pitch-thirds rule compares the attack-frame mean against thirds of 105 or 68, with no second
    negation.
  - The docstring states the convention, removes the ADR-045 D5 text, and gives a two-team example.
  - **Exact relations to the frozen oracle** (on seeded continuous fixtures, which have no exact value ties):
    - attacks +x: `vertical` equals the oracle's. `horizontal` equals the oracle's with L↔R and LC↔RC swapped.
    - attacks −x: `vertical` equals the oracle's, except where the pitch-thirds branch fires (D-tree).
      `horizontal` equals the oracle's on every path except the equal-frequency fallback (TB-SPEC-53).
    - **The −x lateral fallback.** The fallback bins the ascending rank `r` of `68 − y` as `min(5r // n, 4)` onto
      `R, RC, C, LC, L`. The oracle bins the ascending rank of `y` onto `L, …, R`. The two agree only when the binning
      is symmetric, that is for n = 5 and n = 10 players. For the other counts (n = 3, 4, 6, 7, 8, 9: a red card, or
      a player missing from the tracking data), four players move one lateral level (for n = 3, all three players
      move, by up to two levels). The new
      binning is the mirror image of the +x team's binning, so the mirror invariance holds. A test pins one n = 9
      case. NG1, the ADR, and the CHANGELOG name this input class.
  - **Mirror invariance (exact).** For positions `P` stored as float32 (ADR-106) and their point reflection `R(P)`
    computed in float64, `infer_positions(compute_shape_graph(P), P, -1.0)` equals
    `infer_positions(compute_shape_graph(R(P)), R(P), +1.0)`. The reflection of a float32 value in float64 is exact,
    so both calls see bit-equal attack-frame positions.
    - **Premise (TB-SPEC-56):** `compute_shape_graph(P)` and `compute_shape_graph(R(P))` have the same edge set and
      the same face vertex sets. Reviewer B's probe found no difference on 6000 synthetic frames (continuous and
      cm-quantized). A test pins the premise on the seeded fixtures. Qhull can resolve a cocircular set differently
      after the reflection. So a W5.7 mirror failure on a near-cocircular real frame is a known class, and the
      harness reports whether the two graphs' face sets differ for each failing team-frame.
  - **Detection tests, red at the base first:** D-lat (a 4-4-2 that attacks +x: the high-y full back gets "L"),
    D-eqf (a degenerate graph that uses the fallback), D-tree (a team that attacks −x whose middle sub-graph is a
    tree in the defensive third: "DM", not "AM"), and the `ValueError` for `NaN` and `0.0`.
  - `test_lateral_label_is_pitch_absolute_not_team_relative` (`tests/tracking/test_shape_graph.py:285`) goes red, as
    its docstring intends. It is replaced by the team-relative mirror test. The `add_shape_graph` mirror-registry
    note (`tests/tracking/_mirror_entries/shape_and_structure.py:31`) is updated.
- W5.7 **Real-data role gate (owner ruling R15).** A corpus driver `scripts/validate_shape_graph_roles.py`.
  - It uses the `scripts/_driver.py` `for_each` seam (ADR-052, resume before load), declares `PIPELINE` (ADR-128,
    one unit per match), and registers in `ARTIFACT_DRIVERS` with its registry gate and a driver unit test.
  - **Data (amended r7.3, owner ruling, before any measurement; plan review A12):** every match of the public
    SkillCorner set, `scripts._corpus.PUBLIC_CORPUS["skillcorner"]` (20 matches; it equals
    `tests/_skillcorner_sample.py` `MATCH_IDS`), through pining. The driver reads that constant and keeps no second
    list. The set includes the three handoff matches. The first draft named a 10-match subset. Frames come from `_loader_pining.build_skillcorner_frames`.
    The reference is `match.json` `players[].player_role.acronym`. The driver uses periods 1–2 and every 10th frame,
    the handoff's method.
  - **Two arms per match:**
    - the baseline: the frozen oracle `tests/oracles/shape_graph_v4139.py`, with the direction from the frame label
      (the handoff's variant B);
    - the final: `compute_shape_graphs` on the same frames.
  - **Scoring (the handoff's rule, fixed here):**
    - **Population (A7-SPEC-86), fixed as a predicate:** a player is scored if the `match.json` role acronym is not
      `GK` and not `SUB`, `start_time` is not null, and the player has at least one labelled outfield row in the
      sampled frames. In the SkillCorner roster, `SUB` marks an unused bench player; a substitute who came on keeps
      a real role. This is the handoff's population (`repro_shape_graph_roles.py:112-113`), so the baseline
      comparison uses it too. The memo names any population difference from the handoff.
    - **Vertical** (TB-SPEC-54): take the mode of the player's five-level `vertical` label first, then map it to a
      group (`B` → back; `DM`, `M`, `AM` → mid; `F` → forward). This is the handoff's order
      (`repro_shape_graph_roles.py:116`).
    - **Lateral:** map each `horizontal` label to a side first (`L`, `LC` → L; `R`, `RC` → R; `C` → C), then take the
      mode (`:117-118`).
    - **Ties:** the label seen first in frame order wins. That is the handoff's `Counter.most_common` behaviour, so
      the baseline comparison uses the same rule. The frames are in the converter's chronological order. The memo
      counts the tied players per side.
    - The truth comes from a committed table: role acronym → vertical group, and the side from the acronym's first
      letter (`L` or `R`; otherwise the player is not scored for side).
    - The table covers every acronym in the 20 rosters. An acronym outside the table makes the unit raise
      (fail-closed, AG4).
  - **Pre-registered gate (fixed 2026-11-08, before any measurement; amended r7.1, owner ruling R15).** The gate
    uses the construct-validity rule of this repo, discrimination against a permutation null
    (`docs/howto/construct-validity.md`, gate 2), with no fixed agreement number. One rule applies to each of the
    four cells (vertical and lateral, home and away). Every agreement is pooled over the 20 matches.
    1. **Beats a permutation null.** In each draw, the truth labels are shuffled among the scored players of each
       (match, team) block, and the pooled agreement is computed again. For the lateral cells, only the
       side-scored players are shuffled. The shuffle keeps each team's role mix, so the null controls for it.
       There are 10,000 draws from `numpy.random.default_rng` with a fixed seed, which the metrics record. The
       predicted labels do not change. The cell passes when the observed agreement is above the null's 95th
       percentile (`null_p95`). The metrics also record `p = (1 + #{null ≥ observed}) / (1 + draws)`. The
       package has no within-block label shuffle to reuse. `_permute_groups_by_block`
       (`silly_kicks/_group_metrics.py:124`) moves groups across blocks, which is a different null, and the
       existing ICC nulls (`scripts/validate_gk_decision.py:46`, `validate_corners.py:39`) shuffle the whole
       population. So the driver uses `rng.permutation` per block, inline. It does not add a shared helper,
       because no other caller needs one.
    2. **No regression.** In each cell, the final agreement is ≥ the baseline agreement. This rule catches a
       regression only. It is never the pass criterion on its own, because parity with the old numbers alone
       "re-blesses a defective baseline" (`docs/howto/construct-validity.md:99`).
    3. **Mirror invariance (exact).** For every scored team-frame, the final labels on the frames equal the final
       labels on the point-reflected frames, with `team_attacking_direction` swapped (`ltr` ↔ `rtl`). The harness
       reflects in float64, so the equality is exact. It names each differing team-frame.

    The metrics report, for each cell, the agreement with its Wilson 95 % interval, `n`, `null_p95`, and `p`. These
    are reported, not gated. A failure of rule 1, 2, or 3 stops the cycle and goes to the owner. The rules do not
    change after the measurement.

    Note (TB-SPEC-58): the W5.7 frames are `"ltr"`, so the away team always attacks −x. Under the W5.6 relations,
    the away lateral labels and the home vertical labels equal the baseline's, except for the −x fallback. So in
    those two cells, rule 1 measures the algorithm that R14 does not change, and a failure there points at the
    algorithm, not at the label fix. In the home lateral cell, the baseline is below chance (0/7, 2/11, 1/12 in the
    handoff), so rule 1 is the test that the fix works.
  - **Diagnostics in the memo:**
    - per side, the confusion counts of role acronym × modal vertical label;
    - per side, the number of team-frames where the pitch-thirds branch fires;
    - per side, the number of team-frames that use the lateral equal-frequency fallback, and how many of them have
      n ∉ {5, 10} (TB-SPEC-59), so that a rule-2 trip in a lateral cell can be attributed;
    - the baseline on 1886347, 1899585, and 1925299, compared with the handoff's variant-B numbers. The memo explains
      any difference (the frame path differs from the lakehouse's).

    The memo states the cause of the away vertical gap if the work finds it. A defect found there goes to the owner
    before any fix (R15).
  - **Run:** on the DGX (R22), after the code freeze (section 9), with the expected peak memory stated first (W0.5).
    The results land in `docs/research/shape-graph-positions/` (`metrics.json` and the memo), stamped with the
    scoped code identity (W0.4).

**Acceptance:**
- `compute_shape_graph` (edges, faces in order, stabilities) is bit-identical to the frozen pre-change oracle, on
  every CPU class in section 5.2. The fixtures are seeded production-shape frames, plus collinear points, n < 3,
  duplicate points, exact ties, and both directions.
- The W5.1 speed work lands before W5.6. At that point `infer_positions` (labels) is still bit-identical to the
  oracle in both directions. After W5.6, its labels meet the W5.6 relations against the oracle.
- `add_shape_graph` and `shape_graph_xfns` outputs are bit-identical before and after, on inputs without a `game_id`
  dtype mismatch. A detection test shows the old mismatch (R2).
- `compute_shape_graphs` equals the per-frame public functions called one frame at a time. `graph()` equals the
  retained graph.
- An `assert_subquadratic_growth` guard on `compute_shape_graphs` over frames.
- A corpus-scale parity pass in the W0 driver (all frames of the measured matches, both teams).
- `compute_shape_graphs` raises on mixed directions inside one team-frame. Chunked calls on whole-frame boundaries
  concatenate to one call's result (R16).
- The W5.6 detection tests were red at the base. The W5.6 oracle relations and the mirror invariance hold.
- The W5.7 gate passes on its pre-registered rules, and its memo is committed.

### 4.6 W6 — `detect_line_breaking`: correct keys, independence, array pass (handoff item 6)

**Today:**
- `detect_line_breaking` (`tracking/_line_breaking.py:66`) is public (`silly_kicks.tracking.detect_line_breaking`,
  `__init__.py:392`). Its docstring teaches the private import (`:97`), and lakehouse v2 imports
  `silly_kicks.tracking._line_breaking` (reviewer B).
- With `links=None` it calls `link_actions_to_frames` (`:124`), which is game-blind (W9). It then merges the
  pointers with `actions` on `action_id` **alone** (`:130-145`), then `drop_duplicates("action_id", keep="first")`
  (`:147`). `aid_to_pos` maps each `action_id` to its **last** position (`:180`), and `_flip_by_aid` is keyed the same
  way (`:194`). So a duplicate `action_id` gets the **first** occurrence's geometry, written to the **last**
  occurrence's row, and the first row stays `<NA>`. Reviewer A measured this.
- SPADL `action_id` is `range(len)` per `convert_to_actions` call (`spadl/base.py:164`), and a call normally
  converts one game. So actions concatenated from several games carry duplicate `action_id`s, and a game's rows can
  receive another game's geometry.
- The game-id dtype alignment re-keys the frame dicts with `str(k[0])` and casts `linked["game_id"].astype(str)`
  (`:163-177`). Whether it runs depends on the first linked row's type (`:168`). That is an ADR-019 defect.
- The per-action loop is `for _, row in linked.iterrows()` (`:196`).

**Design (owner ruling R1):**
- W6.1 **Key on the action's identity.** The linking is the W9 game-aware linker. Its pointers carry `action_row`
  and `action_fingerprint` (W9.2), and a caller's `links=` follows the W9.5 rule. Every action-keyed step is positional on
  `action_row`: the link merge, the direction lookup, and the output position. The frame lookup keys on
  (game_id, period_id, frame_id), with the action's `game_id` taken by position. The `drop_duplicates("action_id")`
  goes away. Each input row gets **its own** result at **its own** position, including duplicate `action_id`s
  (VAEP gamestate slots, W9.3). The docstring says so.
- W6.2 **Replace the `str` re-key** with `id_compat` aligned keys (R2). No branch depends on the first row.
- W6.3 **State and test independence.** With the same `frames` (and the same `links`, if given), the output row for
  an action does not depend on which other actions are in the call. Callers may chunk `actions` in any way and
  concatenate, including chunkings that split games. Warnings are not part of this contract: a
  chunk can emit or skip the link-rate-floor or `_is_shootout_only` warning differently from the whole call. The
  docstring says so.
- W6.4 **Replace the `iterrows` loop** with an array pass:
  - vectorised type, NaN-actor, and direction masks;
  - one keyed lookup from (game, period, frame) to the opponent group through the `group_rows` index;
  - a loop only over the actions that need `_straddle_core` (the Ward clustering stays per action).

  Values are bit-identical to the pre-change function on every single-game input with unique `action_id`, including
  every NaN route: NaN actor, non-pass, unresolved direction, and no opponents → 0.

**Acceptance:**
- **Detection first:** a test that fails on the pre-change code for a two-game input, on both `links=None` and
  caller-supplied `links=` (W9 pointers). It asserts each row's own result.
- Bit-identity against the frozen pre-change oracle on seeded fixtures and on the corpus, for single-game
  unique-key inputs.
- A property test: `concat(detect_line_breaking(chunk_i, frames))` equals `detect_line_breaking(actions, frames)` for
  random chunkings, including chunkings that split games. Values are compared, warnings are not.
- The W9 game-split property: two games in one call equal the two games called separately.
- An `assert_subquadratic_growth` guard over actions.
- CHANGELOG note: values change only for multi-game inputs (actions concatenated across converter calls).

### 4.7 W7 — `smooth_frames` and `derive_velocities` speed-up (owner additions R4)

**Today:**
- Both functions sort the frames, group by (game_id, period_id, is_ball, player_id) with
  `groupby(...).groups`, and process each group:
  - `smooth_frames` (`preprocess/_smoothing.py:155`) runs `_smooth_positions_dense` per group: savgol, ema,
    butterworth, or `kalman_ca` positions;
  - `derive_velocities` (`preprocess/_velocity.py:120`) runs `_derive_group_dense` per group.
- `_derive_group_dense` always runs `kalman_ca` on x and y. `kalman_ca` is a Python per-frame loop over 3×3 `@`
  products, with `np.linalg.inv` in the RTS pass (`preprocess/_kalman.py:68-86`).
- The derivative is savgol or finite differences per segment.
- W0 confirms where the time goes before any change.

**Design (the builder chooses under the bit-identity gate; W0 decides what is worth doing):**
- W7.1 **Data-independent covariance reuse.** The covariance recursion (`p`, `pp`, the gain, and the RTS `c_mat`)
  depends only on `dt`, the config, and the observed/missing mask, not on the positions. Memoise it per (dt, config,
  mask) within one call, and share it between x and y and across players. This repeats the same operations on the
  same inputs, so it is bit-identical by construction. The gate still proves it.
- W7.2 **A numba Kalman kernel**, as its own family (W1), with the numpy version kept as the reference. numpy's
  3×3 `@` goes through BLAS, and `np.linalg.inv` through LAPACK, so a hand-written loop is **not** bit-identical by
  default. W7.2 ships only under the §5.2 single rule (every platform we run, R13). Otherwise it is dropped, and the
  measured gain it would have given is recorded (section 5.3).
- W7.3 **Less pandas overhead** in both functions: `groupby(...).groups` → `group_rows` positions, one sort, and no
  per-group `list(idx)`. Bit-identical by construction.
- W7.4 `_ema_per_group` and the butterworth path get the same treatment only where W0 shows a material share. The
  `_butterworth.py` public-to-scripts symbols (`PRIVATE_CONSUMERS.md:47`) keep their signatures and behaviour.

**Acceptance:**
- **`derive_velocities`:** all nine output columns (`vx`, `vy`, `speed`, `accel_x`, `accel_y`, `accel`, `pos_var`,
  `vel_var`, `accel_var`) are bit-identical, together with the plausibility guard's NaNs and warnings.
- **`smooth_frames`:** `x_smoothed`, `y_smoothed`, and `_preprocessed_with` are bit-identical.
- Both hold for every smoothing method (`savgol`, `butterworth`, `ema`, `kalman`) and on every CPU class.
- `test_preprocess_baseline_integrity.py` stays green.

### 4.8 W8 — DFL event period, time, and actor derivation (handoff 2)

**Today (`providers/sportec/parse.py`):**
- `_scan_kickoff_times` (`:881`) records, per recognised GameSection, the first KickOff **with a parseable
  EventTime**, in document order. A KickOff with an unparseable time is skipped, so a later one can win. An
  unrecognised section is logged and skipped.
- `_derive_period_from_kickoffs` (`:945`) picks the period whose kickoff time is the **latest** one ≤ the event
  time. On equal kickoff times, the first recorded wins (`:956-964`). It is not "the largest period"; handoff 2 had
  this wrong.
- `_build_event_row` (`:817-829`) sets `timestamp_seconds` = (event time − period start) as `timedelta`, then
  `round(total_seconds(), 4)`, using Python `round`.
- The parser drops these rows without counting them:
  - an Event with no first child (`:1036`);
  - an Event whose EventTime is missing or unparseable;
  - an Event whose EventTime cannot be compared with the kickoffs (the mixed tz-aware/naive `TypeError`, caught at
    `:1052`);
  - an Event before every kickoff.

  It returns an empty list when no KickOff exists (`:1018`).
- **Actor.** The parser reads the player from `_PLAYER_ATTR_ORDER[event_type]` on the first child (KickOff: the
  **first** nested `Play`) (`:832-845`). The team comes from `player_team_map`, and `team_id_native` from the
  home/away label (`:846-857`).
- **What the bronze stores.**
  - First-child attributes are stored only for event types in `_EVENT_TYPE_PREFIX` (`:858-863`). So an unprefixed
    type's actor attribute is not in the bronze.
  - A nested `Play` is stored under `play_*`, and a later nested `Play` overwrites an earlier one. So the bronze
    keeps the **last** nested `Play`.
- `shape_events_to_native` then fills `team`/`player_id` from set-piece qualifier columns
  (`_resolve_idsse_team_from_qualifiers`, `:1370`). That fill reads `home_team_id_native`/`away_team_id_native`
  (`:1391-1392`) and compares CLU ids with a raw `==` (R2).

**Design (general DFL functionality, AG3):**
- W8.1 **One period rule, two adapters.**
  - A pure internal rule `_dfl_period_rule` works on parsed datetimes and kickoffs collected in document order. It is
    exactly today's semantics.
  - Public adapter:
    `derive_dfl_event_periods(event_time, event_type, kickoff_game_section) -> tuple[pd.DataFrame, DflPeriodReport]`
    in `silly_kicks.providers.sportec`.
  - **Inputs** are index-aligned Series, in document order (the docstring requires it).
    - `event_time` is either ISO-8601 strings, parsed with the parser's exact rule (`datetime.fromisoformat`,
      tz-aware → UTC), or a datetime64 Series (tz-aware → UTC; naive kept naive).
    - The datetime64 path must give results bit-identical to the string path for the same instants (reviewer A).
      Python `datetime` holds microseconds and datetime64 can hold nanoseconds. The adapter converts each value to
      the same Python `datetime`/`timedelta` arithmetic, so `round(total_seconds(), 4)` sees identical operands.
    - **The nanosecond rule:** a datetime64 value is floored to whole microseconds (`.dt.floor("us")`) before the
      conversion. That avoids the pandas "Discarding nonzero nanoseconds" `UserWarning` from `to_pydatetime`.
      Flooring and truncation agree for post-1970 instants. The docstring states the rule. Any discarded
      sub-microsecond part is below the 4-decimal rounding of `timestamp_seconds`.
    - **String parsing follows the running Python version, as the parser does today.** `datetime.fromisoformat` on
      Python 3.10 (in the CI matrix, `ci.yml:56`) rejects a `Z` suffix and more than 6 fractional digits. Python
      3.11+ accepts more forms. The adapter does not change this, so the parser stays bit-identical on each Python.
      A string that the running Python rejects is an "unparseable time" row. The docstring documents the version
      behaviour.
    - The string-vs-datetime64 bit-equality test uses only inputs that every CI Python accepts: explicit `±hh:mm`
      offsets and at most 6 fractional digits. A separate test pins the 3.10 rejections as "unparseable" rows.
  - **Output** columns: `period` (`Int64`) and `timestamp_seconds` (`float64`), index-aligned.
  - **Rows without a period** give `<NA>`/NaN and are counted. This is the repo pattern (`ConversionReport`;
    ADR-051 "unresolved is a `<NA>` value"). The rows are:
    - pre-kickoff. This is routine (warm-up events; `parse.py:950`), so it is counted only and **never warned**.
    - unparseable time. A defect: counted and warned.
    - incomparable time (mixed tz-aware/naive; owner ruling R3). A defect: counted and warned.

    The public adapters warn once per call with the dedicated category `DflEventTimeWarning`, for defects only.
    `parse_dfl_events` emits **no** new warning. It logs the counts at INFO (pre-kickoff) and WARNING (defects)
    through its existing logger, so its observable warnings stay as they are today (reviewer B).
  - An unrecognised GameSection makes the public adapters emit `DflGameSectionWarning`. The parser keeps its
    existing log line. Neither warning subclasses another (AGENTS.md).
  - **`DflPeriodReport`** (frozen) holds:
    - `kickoff_times: dict[int, datetime]`;
    - `n_pre_kickoff`, `n_unparseable_time`, `n_incomparable_time`;
    - `unrecognized_sections: tuple[str, ...]`;
    - `skipped_kickoffs: tuple[int, ...]`: periods where a later KickOff of the same period was ignored, because the
      first parseable one wins.
  - **No KickOff at all** is the whole-input case. It returns all-`<NA>`, and the report has empty `kickoff_times`.
    The parser keeps its "return no rows" behaviour on top of that.
- W8.2 **One actor rule, two adapters.**
  - A pure internal rule `_dfl_actor_rule(event_type, actor_attrs: Mapping[str, str]) -> str` applies
    `_PLAYER_ATTR_ORDER` (default `["Player"]`) to the "search element" attributes.
  - The search element is the parser's (`parse.py:835-845`):
    - for KickOff, the **first** nested `Play` whenever one exists, **even if it has no `Player`** (the result is
      then `""`/`"unknown"`);
    - for KickOff, the KickOff element's own attributes **only when no nested `Play` exists**;
    - for every other type, the first child.
  - The parser calls it with the XML attributes. That is exactly today's result, including unprefixed types and
    multiple nested `Play`s.
  - Public adapter: `resolve_dfl_event_actors(events, match_info) -> pd.DataFrame`, with columns `player_id`,
    `team`, and `team_id_native`.
    - It reads the actor attributes from the **DFL actor columns** of the public column contract (W8.3).
    - It maps the player to `"home"`/`"away"`/`"unknown"` through `match_info.player_team_map`, and to
      `team_id_native` through `match_info.home_team_id`/`away_team_id`.
    - A missing column or an empty value counts as "attribute absent", which is the parser's `get(attr, "")`.
- W8.3 **A public, documented column contract**, so any DFL consumer can conform:
  - `dfl_event_column(event_type: str, attribute: str, *, nested: str | None = None) -> str`;
  - the read-only maps `DFL_EVENT_TYPE_PREFIX` and `DFL_NESTED_PREFIX`;
  - `dfl_actor_columns(event_type) -> tuple[str, ...]`: the ordered actor columns for one event type.

  The contract defines a column for **every** DFL event type and for the KickOff actor:
  - **Naming.** A type in `_EVENT_TYPE_PREFIX` uses its prefix. Any other DFL tag uses `snake_case(tag)`, through the
    parser's own `_to_snake_case`. A nested element uses the parser's `_NESTED_PREFIX_MAP` prefix. So
    `dfl_event_column(t, a)` is `f"{prefix(t)}_{snake(a)}"`, and with `nested=n` it is `f"{nested_prefix(n)}_{snake(a)}"`.
    That is the bronze rule, so `dfl_event_column("KickOff", "Player", nested="Play")` is `play_player`.
  - For types that the silly-kicks bronze does not emit, the contract still defines the column name, so a consumer
    bronze can carry it. The silly-kicks bronze is unchanged (NG4).
  - **KickOff actor columns** are contract-only and explicitly named, not produced by `dfl_event_column`. The bronze
    `play_*` columns hold the **last** nested `Play`, so they cannot carry the first-Play rule:
    - `kickoff_play_present` (boolean): a nested `Play` exists. **Required** whenever `events` has a KickOff row; a
      missing column raises `ValueError` (a whole-input contract break);
    - `kickoff_first_play_player`: the FIRST nested `Play`'s `Player`;
    - `kickoff_player`: the KickOff element's own `Player`, read only when `kickoff_play_present` is False.

    `dfl_actor_columns("KickOff")` returns those three in that order and documents the fallback.

  The silly-kicks bronze already carries the resolved `player_id`/`team`, so it never needs this resolver. The
  docstring says so. That avoids its "last nested Play" storage. The handoff's v2 name `kick_off_game_section`
  versus the contract's `kickoff_game_section` shows why the contract must be explicit.
- W8.4 **A convenience composite:** `complete_dfl_event_bronze(events, match_info) -> tuple[pd.DataFrame, DflPeriodReport]`.
  - It adds `period`, `timestamp_seconds`, `player_id`, `team`, `team_id_native`, and the match-level columns that
    the qualifier fill needs: `home_team_id_native` and `away_team_id_native` (reviewer A), plus
    `competition_native_id` and `season_native_id` where `match_info` has them.
  - The input frame has `event_time`, `event_type`, `kickoff_game_section`, and the contract's actor columns.
  - The output is ready for `shape_events_to_native`. Rows with a `<NA>` period are **kept**, and the caller decides.
    The docstring shows the one-line drop that reproduces the parser.
- W8.5 **The parser is single-sourced.**
  - One XML pass collects the rows, with raw `event_time`, as today.
  - Each row's actor comes from `_dfl_actor_rule` on the XML attributes.
  - The periods and timestamps come from `_dfl_period_rule` through the same code path as W8.1.
  - The parser then drops exactly the rows it drops today, so the bronze is bit-identical (NG4).
  - The private helpers `_scan_kickoff_times` and `_derive_period_from_kickoffs`, the inline time block in
    `_build_event_row`, and the second XML pass are removed.
  - The parser logs the report counts that it drops silently today, and the no-first-child count.
- W8.6 **The qualifier fill changes only under R2.** `_resolve_idsse_team_from_qualifiers` compares CLU ids through
  `id_compat`. Values change only where the old raw `==` silently missed because the dtypes differed. Everything
  else in `shape_events_to_native` is unchanged.

**Acceptance:**
- **Bronze bit-identity:** `parse_dfl_events` output before and after, on every DFL match in the corpus (idsse via
  pining): all columns, row set, row order, and dtypes.
- **Corpus census** in the W0 memo: how many events have an unprefixed type, a KickOff with more than one nested
  `Play`, an attribute-less `Play`, a skipped KickOff, an incomparable time, or no first child.
- **W8.1 unit tests:**
  - the latest-kickoff rule, including non-monotone kickoff times and equal kickoff times;
  - the two-pass case: a first-half-time event after the second-half KickOff in row order is period 1;
  - pre-kickoff, unparseable, and incomparable rows → `<NA>`, counted, and warned;
  - an unrecognised section → warned and reported;
  - a KickOff with an unparseable time is skipped, and the next one wins;
  - Python `round` semantics at the 4th decimal, with a half-way probe;
  - the string path and the datetime64 path agree bit for bit (UTC, a non-UTC offset, sub-microsecond input).
- **W8.1 corpus pass:** on every DFL match, the datetime64 path (the bronze `event_time` strings converted with
  `pd.to_datetime(utc=True)`) equals the string path bit for bit (reviewer A).
- **W8.2:** parity with the parser's actor columns on the corpus. Unit cases cover:
  - every row of `_PLAYER_ATTR_ORDER` and an unprefixed type;
  - a KickOff with two nested `Play`s, a KickOff without a `Play`, and a `Play` without a `Player`;
  - a missing `kickoff_play_present` column (raises);
  - an unknown player.
- **R2 detection test** for the qualifier fill (an int vs str CLU id dtype mismatch).
- **End-to-end:** XML → `parse_dfl_events` → `shape_events_to_native` → convert, compared with consumer-shaped input
  (contract columns, no derived columns) → `complete_dfl_event_bronze` → drop `<NA>` period →
  `shape_events_to_native` → convert. Both give identical SPADL actions. This closes the handoff's 18–27-actions
  failure mode.

### 4.9 W9 — Multi-game linking at the root (owner ruling R5)

**Verified defect (author reproduction 2026-11-08 on `af03c4b`, after reviewer A's r2 measurement).** The input was
two games with the same period and frame ids, where game 8's clock runs 2 s later, and actions with `action_id` 0, 1
in each game. `link_actions_to_frames` gave:
- **Frames dropped:** the frame index keeps 100 of the 200 distinct (game, period, frame) rows, because it dedups on
  `["period_id", "frame_id"]` (`tracking/utils.py:414-418`).
- **Wrong game, wrong frame:** game 8's actions link to game 7's frames 60 and 80, not to game 8's frames 10 and 30.
  The linking groups and merges on period and time only (`:426-445`).
- **The pointers carry no `game_id`** (`:463-470`). Consumers join them back on `action_id` alone. The usual
  `merge(on="action_id")` turns 4 actions into 8 rows: each action is paired with both games' pointers.
- `_count_candidates_within_tolerance` is game-blind too (`:1262-1268`).

The repo already treats multi-game frames as supported: `smooth_frames`, `derive_velocities`, and
`interpolate_frames` key on `game_id` "so a two-game frame never … across the game boundary" (TF-65 §4.1a).
`game_id` is a required column in both `TRACKING_FRAMES_COLUMNS` (`tracking/schema.py:11`) and `SPADL_COLUMNS`
(`spadl/schema.py:10`). The linker is the gap.

**Callers (AGENTS.md: a spec that changes a public seam enumerates every caller).** These are the package call sites
of `link_actions_to_frames` / `link_actions_to_frames_elastic` at `af03c4b`:

| File | Lines |
|---|---|
| `tracking/features.py` | 285, 1017, 1269, 1507, 1583, 1764, 2262, 2445, 2620, 3465, 3751, 4202, 4532, 5208, 5432, 5570, 5729, 6250, 6563, 7074, 7190, 7235 |
| `atomic/tracking/features.py` | 439, 1132, 1248, 1480 |
| `calibration/_features.py` | 202, 326 |
| `calibration/_carrier_objective.py` | 84 |
| `gk_decision/_reconstruct.py` | 128 |
| `keeper_identity.py` | 959 |
| `restdefense/_windows.py` | 110 |
| `tracking/_gk_resolve.py` | 156 |
| `tracking/_kernels.py` | 874, 960 |
| `tracking/_line_breaking.py` | 124 |
| `tracking/_off_ball_runs.py` | 329 |
| `tracking/_press_commitment.py` | 111 |
| `tracking/_receiver.py` | 598 |
| `tracking/_run_features.py` | 284 |
| `tracking/_xcross_attempt.py` | 1019, 1075 |
| `tracking/_xshot_occurrence.py` | 1131, 1196 |
| `tracking/defensive_credit/_orchestration.py` | 96 |
| `tracking/utils.py` | 1307 |

The plan must cover three more things, with line evidence for both sides of each join:
- every function that **accepts** caller-supplied `links=` (pre-linking, ADR-008), whether or not it calls the
  linker itself;
- every join of pointers to actions;
- the 197 test and script files that reference the linker. They are checked, not necessarily changed.

**Consumer census (plan Task 5 step 1; 2026-11-11).** The table lists every member of the W9.6 population (287
members). A scratch script generated it from the W9.6 call graph (`tests/invariants/_link_population.py`) on the
Task 4 working tree. The line numbers are the line numbers of that tree. The builder reads each site again when the
builder migrates its row, and then ticks the row.

- **Pointer source:** `links=` (the member has a `links` parameter), `linker` (it calls a linker), `ctx` (it calls
  `_resolve_action_frame_context`), `helper` (it gets the pointers only through another member), `global` (a default
  xfn list), or `class`.
- **Linker / ctx call:** `link@N` is a call of `link_actions_to_frames` at line N, `link_elastic@N` is a call of the
  elastic linker, and `ctx@N` is a call of `_resolve_action_frame_context`.
- **Join columns:** `kind@lines` is one call in the member's own body, not in a nested function. The action column
  holds the joins keyed on `action_id`: pointer to action, and a result back to its action. The frame column holds
  the joins and the frame lookups keyed on `frame_id`, with or without `period_id`. `-` means that the member has no
  such call. It then reaches the pointers through the members in the last column.
- The five link-identity functions are link machinery (owner ruling 2026-11-11, `LINK_MACHINERY`). They are not
  members, so they have no row.

| Done | Member | File:lines | Pointer source | Linker / ctx call | Pointer or result -> action join | Pointer -> frame join | Forwards to (members) |
|---|---|---|---|---|---|---|---|
| [ ] | `_main` | `scripts/_das_path_timing.py:166-193` | helper | - | - | - | time_add_das_and_xfns |
| [ ] | `time_add_das_and_xfns` | `scripts/_das_path_timing.py:133-163` | linker | link@146 | - | - | time_add_das_and_xfns.‹l›._add, time_add_das_and_xfns.‹l›._xfn, add_das |
| [ ] | `time_add_das_and_xfns.‹l›._add` | `scripts/_das_path_timing.py:149-150` | helper | - | - | - | add_das |
| [ ] | `time_add_das_and_xfns.‹l›._xfn` | `scripts/_das_path_timing.py:152-153` | helper | - | - | - | - |
| [ ] | `receiver_failed_pass_accuracy` | `scripts/_receiver_validation.py:126-164` | links=+linker | link@129 | get@137 | groupby@133, zip@132, set@132 | trajectory_weak_labels |
| [ ] | `trajectory_weak_labels` | `scripts/_receiver_validation.py:78-123` | links=+linker | link@89 | - | groupby@92, zip@91, set@91 | - |
| [ ] | `extract_played_passes` | `scripts/_rq_corpus.py:46-112` | links=+linker | link@59 | set_index@60 | groupby@68, map@60 | - |
| [ ] | `call_aggregator` | `scripts/_sb_battery.py:292-299` | links= | - | - | - | generic |
| [ ] | `defensive_credit` | `scripts/_sb_battery.py:121-140` | helper | - | - | - | defensive_credit.‹l›.call |
| [ ] | `defensive_credit.‹l›.call` | `scripts/_sb_battery.py:129-138` | links= | - | - | - | - |
| [ ] | `generic` | `scripts/_sb_battery.py:50-62` | helper | - | - | - | generic.‹l›.call |
| [ ] | `generic.‹l›.call` | `scripts/_sb_battery.py:54-60` | links= | - | - | - | - |
| [ ] | `gradientsports_player_ids` | `scripts/_sb_battery.py:161-198` | helper | - | - | - | gradientsports_player_ids.‹l›.call |
| [ ] | `gradientsports_player_ids.‹l›.call` | `scripts/_sb_battery.py:170-196` | links= | - | - | - | - |
| [ ] | `pre_shot_gk_angle` | `scripts/_sb_battery.py:110-118` | helper | - | - | - | pre_shot_gk_angle.‹l›.call |
| [ ] | `pre_shot_gk_angle.‹l›.call` | `scripts/_sb_battery.py:115-116` | links= | - | - | - | - |
| [ ] | `pre_shot_gk_position` | `scripts/_sb_battery.py:152-158` | helper | - | - | - | pre_shot_gk_position.‹l›.call |
| [ ] | `pre_shot_gk_position.‹l›.call` | `scripts/_sb_battery.py:155-156` | links= | - | - | - | - |
| [ ] | `run_add_star_battery` | `scripts/_sb_battery.py:302-318` | links= | - | - | - | call_aggregator |
| [ ] | `sync_score` | `scripts/_sb_battery.py:143-149` | helper | - | - | - | sync_score.‹l›.call |
| [ ] | `sync_score.‹l›.call` | `scripts/_sb_battery.py:146-147` | links= | - | - | - | - |
| [ ] | `visible_area_coverage` | `scripts/_sb_battery.py:201-216` | helper | - | - | - | visible_area_coverage.‹l›.call |
| [ ] | `visible_area_coverage.‹l›.call` | `scripts/_sb_battery.py:212-214` | links= | - | - | - | - |
| [ ] | `with_pressure_methods` | `scripts/_sb_battery.py:219-232` | helper | - | - | - | with_pressure_methods.‹l›.call |
| [ ] | `with_pressure_methods.‹l›.call` | `scripts/_sb_battery.py:226-230` | links= | - | - | - | - |
| [ ] | `with_xt` | `scripts/_sb_battery.py:65-75` | helper | - | - | - | with_xt.‹l›.call |
| [ ] | `with_xt.‹l›.call` | `scripts/_sb_battery.py:69-73` | links= | - | - | - | - |
| [ ] | `with_xt_keyword` | `scripts/_sb_battery.py:78-96` | helper | - | - | - | with_xt_keyword.‹l›.call |
| [ ] | `with_xt_keyword.‹l›.call` | `scripts/_sb_battery.py:90-94` | links= | - | - | - | - |
| [ ] | `main` | `scripts/build_rq_pass_scores.py:129-192` | helper | - | - | - | score_match |
| [ ] | `score_match` | `scripts/build_rq_pass_scores.py:73-126` | linker | link@79 | - | groupby@86 | extract_played_passes, pitch_control_at_target |
| [ ] | `_pass_b` | `scripts/derive_coordination_params.py:740-778` | helper | - | - | - | _pass_b.‹l›.work, pass_b_match |
| [ ] | `_pass_b.‹l›.work` | `scripts/derive_coordination_params.py:756-757` | helper | - | - | - | pass_b_match |
| [ ] | `main` | `scripts/derive_coordination_params.py:1766-1782` | helper | - | - | - | _pass_b |
| [ ] | `pass_b_match` | `scripts/derive_coordination_params.py:671-728` | helper | - | - | - | _build_coordination_signals, build_coordination_signals |
| [ ] | `_owner_eval_receiver` | `scripts/hf_jobs/headtohead.py:704-719` | helper | - | - | - | extract_candidate_rows |
| [ ] | `main` | `scripts/measure_cover_shadow_argmax_agreement.py:282-409` | helper | - | - | - | measure_match |
| [ ] | `measure_match` | `scripts/measure_cover_shadow_argmax_agreement.py:110-210` | linker | link@118 | set_index@119 | groupby@120 | - |
| [ ] | `_feat_receiver` | `scripts/measure_f1b_feature_delta.py:224-258` | linker | link@239 | get@244 | zip@240, set@240 | - |
| [ ] | `_linker_inputs` | `scripts/measure_tracking_compute.py:359-362` | linker | link@362 | - | - | - |
| [ ] | `_memory_work` | `scripts/measure_tracking_compute.py:1184-1190` | helper | - | - | - | step_plan |
| [ ] | `_prereq_work` | `scripts/measure_tracking_compute.py:1153-1177` | helper | - | - | - | _linker_inputs, slice_pointers |
| [ ] | `_probe_main` | `scripts/measure_tracking_compute.py:1580-1590` | helper | - | - | - | probe_unit |
| [ ] | `_profile_work` | `scripts/measure_tracking_compute.py:1193-1212` | helper | - | - | - | step_plan |
| [ ] | `main` | `scripts/measure_tracking_compute.py:1706-1715` | helper | - | - | - | _probe_main |
| [ ] | `probe_unit` | `scripts/measure_tracking_compute.py:917-933` | helper | - | - | - | step_plan |
| [ ] | `slice_pointers` | `scripts/measure_tracking_compute.py:618-621` | helper | - | - | - | _linker_inputs |
| [ ] | `step_plan` | `scripts/measure_tracking_compute.py:814-900` | helper | - | - | - | _linker_inputs, detect_line_breaking |
| [ ] | `_convert_and_write` | `scripts/probe_action_context_baselines.py:659-723` | helper | - | - | - | _verify_link_rate |
| [ ] | `_process_lakehouse_provider` | `scripts/probe_action_context_baselines.py:601-635` | helper | - | - | - | _convert_and_write |
| [ ] | `_process_skillcorner` | `scripts/probe_action_context_baselines.py:638-656` | helper | - | - | - | _convert_and_write |
| [ ] | `_verify_link_rate` | `scripts/probe_action_context_baselines.py:571-598` | linker | link@593 | - | - | - |
| [ ] | `main` | `scripts/probe_action_context_baselines.py:804-863` | helper | - | - | - | _process_lakehouse_provider, _process_skillcorner |
| [ ] | `_enrich_from_slim_parquet` | `scripts/probe_preprocess_baseline.py:188-306` | linker | link@273 | - | - | add_pre_shot_gk_context, pre_shot_gk_angle_off_goal_line, pre_shot_gk_angle_to_shot_trajectory |
| [ ] | `main` | `scripts/probe_preprocess_baseline.py:359-421` | helper | - | - | - | _enrich_from_slim_parquet |
| [ ] | `_deploy_work` | `scripts/train_receiver_model.py:571-574` | helper | - | - | - | _deployment_counts_for_match |
| [ ] | `_deployment_counts_for_match` | `scripts/train_receiver_model.py:304-328` | linker | link@309 | - | - | receiver_failed_pass_accuracy |
| [ ] | `_rows_work` | `scripts/train_receiver_model.py:448-456` | helper | - | - | - | _rows_work.‹l›.work, extract_candidate_rows |
| [ ] | `_rows_work.‹l›.work` | `scripts/train_receiver_model.py:449-454` | helper | - | - | - | extract_candidate_rows |
| [ ] | `deployment_gate` | `scripts/train_receiver_model.py:292-301` | links= | - | - | - | receiver_failed_pass_accuracy |
| [ ] | `extract_candidate_rows` | `scripts/train_receiver_model.py:103-182` | linker | link@122 | get@132 | groupby@127, zip@126, set@126 | - |
| [ ] | `_match_records` | `scripts/tune_structural_pass_sigma.py:28-68` | linker | link@39 | get@45, set_index@40 | groupby@42 | - |
| [ ] | `main` | `scripts/tune_structural_pass_sigma.py:85-119` | helper | - | - | - | _match_records |
| [ ] | `_bench_match` | `scripts/validate_das_native_parity.py:1044-1109` | helper | - | - | - | _scored_frames |
| [ ] | `_measure_match` | `scripts/validate_das_native_parity.py:416-498` | helper | - | - | - | _scored_frames |
| [ ] | `_scored_frames` | `scripts/validate_das_native_parity.py:342-370` | linker | link@363 | merge@364 | merge@370, isin@356 | - |
| [ ] | `main` | `scripts/validate_das_native_parity.py:1269-1438` | helper | - | - | - | run_benchmark, run_corpus |
| [ ] | `run_benchmark` | `scripts/validate_das_native_parity.py:1189-1246` | helper | - | - | - | _bench_match |
| [ ] | `run_corpus` | `scripts/validate_das_native_parity.py:768-833` | helper | - | - | - | _measure_match, run_corpus.‹l›._work |
| [ ] | `run_corpus.‹l›._work` | `scripts/validate_das_native_parity.py:803-804` | helper | - | - | - | _measure_match |
| [ ] | `main` | `scripts/validate_sb360_licensed_corpus.py:220-279` | helper | - | - | - | measure_match |
| [ ] | `measure_match` | `scripts/validate_sb360_licensed_corpus.py:84-185` | linker | link@107 | - | - | run_add_star_battery, add_visible_area_coverage, add_action_context |
| [ ] | `add_pre_shot_gk_context` | `atomic/spadl/utils.py:785-993` | links= | - | - | - | add_pre_shot_gk_angle, add_pre_shot_gk_position |
| [ ] | `_aggregate_defensive_credit` | `atomic/tracking/features.py:378-406` | links= | - | - | - | compute_defensive_credits, _rollup_defending_aggregate |
| [ ] | `actor_arc_length_pre_window` | `atomic/tracking/features.py:1070-1086` | helper | - | - | - | - |
| [ ] | `actor_speed` | `atomic/tracking/features.py:693-706` | ctx | ctx@705 | - | - | _resolve_action_frame_context |
| [ ] | `add_action_context` | `atomic/tracking/features.py:755-790` | links=+ctx | ctx@776 | merge@789, set_index@786 | - | _resolve_action_frame_context |
| [ ] | `add_actor_pre_window` | `atomic/tracking/features.py:1109-1137` | links=+linker | link@1132 | merge@1136, set_index@1133 | - | - |
| [ ] | `add_cover_shadows` | `atomic/tracking/features.py:1360-1410` | links= | - | - | - | add_cover_shadows |
| [ ] | `add_defensive_credit` | `atomic/tracking/features.py:409-455` | links=+linker | link@439 | merge@454, set_index@453, drop_duplicates@453 | - | _aggregate_defensive_credit |
| [ ] | `add_off_ball_run_values` | `atomic/tracking/features.py:588-636` | links= | - | - | - | add_off_ball_run_values |
| [ ] | `add_packing` | `atomic/tracking/features.py:486-531` | links= | - | - | - | add_packing |
| [ ] | `add_pitch_control` | `atomic/tracking/features.py:1305-1324` | links= | - | - | - | pitch_control_at_target |
| [ ] | `add_pre_shot_gk_angle` | `atomic/tracking/features.py:1018-1045` | links=+ctx | ctx@1040 | - | - | _resolve_action_frame_context |
| [ ] | `add_pre_shot_gk_position` | `atomic/tracking/features.py:906-964` | links=+ctx | ctx@955 | merge@963, set_index@960 | - | _resolve_action_frame_context |
| [ ] | `add_press_commitment` | `atomic/tracking/features.py:1414-1440` | links= | - | - | - | add_press_commitment |
| [ ] | `add_pressure_on_actor` | `atomic/tracking/features.py:1218-1253` | links=+linker | link@1248 | merge@1252, set_index@1249 | - | pressure_on_actor |
| [ ] | `add_shot_goalmouth` | `atomic/tracking/features.py:1449-1484` | links=+linker | link@1480 | merge@1483, set_index@1482 | - | compute_shot_goalmouth |
| [ ] | `add_structural_pass` | `atomic/tracking/features.py:156-173` | links= | - | - | - | add_structural_pass |
| [ ] | `atomic_pitch_control_xfns` | `atomic/tracking/features.py:1327-1348` | helper | - | - | - | atomic_pitch_control_xfns.‹l›._pc_helper, pitch_control_at_target |
| [ ] | `atomic_pitch_control_xfns.‹l›._pc_helper` | `atomic/tracking/features.py:1344-1345` | helper | - | - | - | pitch_control_at_target |
| [ ] | `compute_defensive_credits` | `atomic/tracking/features.py:334-375` | links= | - | - | - | compute_defensive_credits |
| [ ] | `defenders_in_triangle_to_goal` | `atomic/tracking/features.py:735-751` | ctx | ctx@750 | - | - | _resolve_action_frame_context |
| [ ] | `nearest_defender_distance` | `atomic/tracking/features.py:677-690` | ctx | ctx@689 | - | - | _resolve_action_frame_context |
| [ ] | `off_ball_run_value_xfns` | `atomic/tracking/features.py:639-674` | helper | - | - | - | off_ball_run_value_xfns.‹l›._atomic_transformer, off_ball_run_value_xfns |
| [ ] | `off_ball_run_value_xfns.‹l›._atomic_transformer` | `atomic/tracking/features.py:663-670` | helper | - | - | - | - |
| [ ] | `packing_xfns` | `atomic/tracking/features.py:534-559` | helper | - | - | - | packing_xfns.‹l›._atomic_transformer, packing_xfns |
| [ ] | `packing_xfns.‹l›._atomic_transformer` | `atomic/tracking/features.py:553-555` | helper | - | - | - | - |
| [ ] | `pitch_control_at_target` | `atomic/tracking/features.py:1264-1301` | links= | - | - | - | pitch_control_at_target |
| [ ] | `pre_shot_gk_angle_off_goal_line` | `atomic/tracking/features.py:999-1014` | ctx | ctx@1012 | - | - | _resolve_action_frame_context |
| [ ] | `pre_shot_gk_angle_to_shot_trajectory` | `atomic/tracking/features.py:980-996` | ctx | ctx@994 | - | - | _resolve_action_frame_context |
| [ ] | `pre_shot_gk_distance_to_goal` | `atomic/tracking/features.py:859-879` | ctx | ctx@877 | - | - | _resolve_action_frame_context |
| [ ] | `pre_shot_gk_distance_to_shot` | `atomic/tracking/features.py:882-902` | ctx | ctx@900 | - | - | _resolve_action_frame_context |
| [ ] | `pre_shot_gk_x` | `atomic/tracking/features.py:806-833` | ctx | ctx@831 | - | - | _resolve_action_frame_context |
| [ ] | `pre_shot_gk_y` | `atomic/tracking/features.py:836-856` | ctx | ctx@854 | - | - | _resolve_action_frame_context |
| [ ] | `pressure_on_actor` | `atomic/tracking/features.py:1148-1214` | links=+ctx | ctx@1176, ctx@1180, ctx@1201 | - | - | _resolve_action_frame_context |
| [ ] | `receiver_zone_density` | `atomic/tracking/features.py:709-732` | ctx | ctx@729 | - | - | _resolve_action_frame_context |
| [ ] | `structural_pass_xfns` | `atomic/tracking/features.py:176-197` | helper | - | - | - | structural_pass_xfns.‹l›._atomic_transformer, structural_pass_xfns |
| [ ] | `structural_pass_xfns.‹l›._atomic_transformer` | `atomic/tracking/features.py:191-193` | helper | - | - | - | - |
| [ ] | `xt_xfns` | `atomic/vaep/features.py:459-515` | helper | - | - | - | xt_xfns.‹l›._xt |
| [ ] | `xt_xfns.‹l›._xt` | `atomic/vaep/features.py:500-512` | helper | - | - | - | - |
| [ ] | `CarrierAccuracyObjective.evaluate` | `calibration/_carrier_objective.py:175-221` | helper | - | - | - | _prepare_match |
| [ ] | `_match_accuracy` | `calibration/_carrier_objective.py:142-148` | helper | - | - | - | _prepare_match |
| [ ] | `_prepare_match` | `calibration/_carrier_objective.py:62-99` | linker | link@84 | isin@86 | - | - |
| [ ] | `_compute_das` | `calibration/_features.py:113-143` | links= | - | merge@139 | merge@141 | add_das |
| [ ] | `enrich_full` | `calibration/_features.py:281-363` | linker | link@326 | - | - | _compute_das, add_pre_shot_gk_context, add_action_context, add_actor_pre_window, add_cover_shadows, add_defensive_line, +5 |
| [ ] | `enrich_invariant` | `calibration/_features.py:146-234` | linker | link@202 | - | - | _compute_das, add_pre_shot_gk_context, add_action_context, add_actor_pre_window, add_cover_shadows, add_defensive_line, +5 |
| [ ] | `patch_trial_columns` | `calibration/_features.py:237-278` | links= | - | - | - | add_pressure_on_actor |
| [ ] | `compute_team_coordination` | `coordination/_compute.py:2306-2341` | links= | - | - | - | build_coordination_signals, possession_windows_from_actions |
| [ ] | `_build_coordination_signals` | `coordination/_signals.py:725-832` | links= | - | - | - | - |
| [ ] | `build_coordination_signals` | `coordination/_signals.py:692-722` | links= | - | - | - | _build_coordination_signals |
| [ ] | `possession_windows_from_actions` | `coordination/_windows.py:212-305` | links= | - | - | zip@250 | - |
| [ ] | `ReconstructedOptionSet.__init__` | `gk_decision/_reconstruct.py:80-106` | links= | - | - | - | - |
| [ ] | `ReconstructedOptionSet._frame_and_anchors_action_ltr` | `gk_decision/_reconstruct.py:140-200` | helper | - | - | - | ReconstructedOptionSet._frame_id_for |
| [ ] | `ReconstructedOptionSet._frame_id_for` | `gk_decision/_reconstruct.py:121-130` | linker | link@128 | - | - | - |
| [ ] | `ReconstructedOptionSet._rows_for_decision` | `gk_decision/_reconstruct.py:204-262` | helper | - | - | - | ReconstructedOptionSet._frame_and_anchors_action_ltr |
| [ ] | `ReconstructedOptionSet.option_rows` | `gk_decision/_reconstruct.py:264-282` | helper | - | - | - | ReconstructedOptionSet._rows_for_decision |
| [ ] | `_resolve_from_native` | `keeper_identity.py:914-1063` | linker | link@959 | - | - | acting_gk_from_frames, defending_gk_from_frames |
| [ ] | `resolve_keeper_identities` | `keeper_identity.py:329-368` | helper | - | - | - | _resolve_from_native |
| [ ] | `_deterrent` | `restdefense/_arms.py:174-242` | links= | - | - | - | select_rest_defense_samples |
| [ ] | `rest_defense_gk_deterrent` | `restdefense/_arms.py:100-138` | links= | - | - | - | _deterrent |
| [ ] | `rest_defense_outfield_deterrent` | `restdefense/_arms.py:53-97` | links= | - | - | - | _deterrent |
| [ ] | `compute_rest_defense` | `restdefense/_compute.py:264-360` | links= | - | - | - | select_rest_defense_samples |
| [ ] | `select_rest_defense_samples` | `restdefense/_windows.py:71-159` | links=+linker | link@110 | merge@112, align_join_keys@111 | - | - |
| [ ] | `add_pre_shot_gk_context` | `spadl/utils.py:512-756` | links= | - | - | - | defending_gk_from_frames, add_pre_shot_gk_angle, add_pre_shot_gk_position |
| [ ] | `add_restart_coordinates` | `spadl/utils.py:772-854` | links= | - | - | - | resolve_restart_geometry |
| [ ] | `compute_territorial_defense` | `territorial_defense/_compute.py:53-115` | links= | - | - | - | - |
| [ ] | `_side_y` | `tracking/_gk_geometry.py:317-326` | links= | - | - | - | _tracking_ball_xy |
| [ ] | `_throwin_x` | `tracking/_gk_geometry.py:329-336` | links= | - | - | - | _tracking_ball_xy |
| [ ] | `_tracking_ball_xy` | `tracking/_gk_geometry.py:271-314` | links= | - | - | groupby@289 | resolve_frame_ids_by_position |
| [ ] | `_tracking_gk_xy` | `tracking/_gk_geometry.py:119-176` | links= | - | - | groupby@135 | resolve_frame_ids_by_position |
| [ ] | `_tracking_gk_xy_detected` | `tracking/_gk_geometry.py:179-256` | links= | - | - | - | - |
| [ ] | `resolve_restart_geometry` | `tracking/_gk_geometry.py:339-492` | links= | - | - | - | _side_y, _throwin_x, _tracking_ball_xy, _tracking_gk_xy, _tracking_gk_xy_detected |
| [ ] | `_gk_from_frames_linked` | `tracking/_gk_resolve.py:128-212` | links=+linker | link@156 | drop_duplicates@201, merge@164-168 | merge@174-181 | - |
| [ ] | `acting_gk_from_frames` | `tracking/_gk_resolve.py:259-342` | links= | - | - | - | _gk_from_frames_linked |
| [ ] | `defending_gk_from_frames` | `tracking/_gk_resolve.py:215-256` | links= | - | - | - | _gk_from_frames_linked |
| [ ] | `_defensive_line_at_actions` | `tracking/_kernels.py:826-930` | links=+linker | link@874 | merge@880-884 | merge@893-899, align_join_keys@890 | - |
| [ ] | `_packing_at_actions` | `tracking/_kernels.py:1046-1132` | links= | - | - | groupby@1095 | resolve_frame_ids_by_position |
| [ ] | `_structural_pass_at_actions` | `tracking/_kernels.py:964-1043` | links= | - | - | groupby@1006 | resolve_frame_ids_by_position |
| [ ] | `resolve_frame_ids_by_position` | `tracking/_kernels.py:933-961` | links=+linker | link@960 | set_index@961, reindex@957, set_index@956, drop_duplicates@956 | - | - |
| [ ] | `detect_line_breaking` | `tracking/_line_breaking.py:66-291` | links=+linker | link@124 | drop_duplicates@147, merge@130-146 | groupby@152 | - |
| [ ] | `_line_break_kernel` | `tracking/_off_ball_runs.py:281-446` | links=+linker | link@329 | drop_duplicates@366, merge@337-341 | groupby@383, merge@357-363, align_join_keys@356 | - |
| [ ] | `compute_press_commitment` | `tracking/_press_commitment.py:75-187` | links=+linker | link@111 | reindex@114-116, set_index@114-115, drop_duplicates@114 | groupby@120 | - |
| [ ] | `_link_and_index` | `tracking/_receiver.py:594-607` | links=+linker | link@598 | - | groupby@606, zip@604 | - |
| [ ] | `intended_receiver_positions` | `tracking/_receiver.py:651-692` | links= | - | - | - | _link_and_index |
| [ ] | `resolve_intended_receiver` | `tracking/_receiver.py:610-648` | links= | - | get@635 | - | _link_and_index |
| [ ] | `_call_family` | `tracking/_run_features.py:143-194` | links= | - | - | - | - |
| [ ] | `run_tracking_features` | `tracking/_run_features.py:197-338` | links=+linker | link@284 | - | - | resolve_keeper_identities, _call_family |
| [ ] | `value_off_ball_runs` | `tracking/_run_values.py:441-601` | links= | - | - | - | resolve_frame_ids_by_position |
| [ ] | `compute_shot_goalmouth` | `tracking/_shot_goalmouth.py:696-837` | links= | - | groupby@773 | drop_duplicates@768 | - |
| [ ] | `add_visible_area_coverage` | `tracking/_visibility.py:237-342` | links= | - | - | - | - |
| [ ] | `add_xcross_attempt` | `tracking/_xcross_attempt.py:999-1051` | links=+linker | link@1019 | merge@1048, set_index@1047, drop_duplicates@1046, merge@1039 | set@1023 | - |
| [ ] | `xcross_attempt_xfns` | `tracking/_xcross_attempt.py:1054-1087` | helper | - | - | - | add_xcross_attempt, xcross_attempt_xfns.‹l›._transformer |
| [ ] | `xcross_attempt_xfns.‹l›._transformer` | `tracking/_xcross_attempt.py:1066-1083` | linker | link@1075 | - | - | add_xcross_attempt |
| [ ] | `add_xshot_occurrence` | `tracking/_xshot_occurrence.py:1105-1166` | links=+linker | link@1131 | merge@1160, set_index@1159, drop_duplicates@1158, merge@1146 | set@1135 | - |
| [ ] | `xshot_occurrence_xfns` | `tracking/_xshot_occurrence.py:1169-1208` | helper | - | - | - | add_xshot_occurrence, xshot_occurrence_xfns.‹l›._transformer |
| [ ] | `xshot_occurrence_xfns.‹l›._transformer` | `tracking/_xshot_occurrence.py:1187-1204` | linker | link@1196 | - | - | add_xshot_occurrence |
| [ ] | `_aggregate_defensive_credit` | `tracking/defensive_credit/_orchestration.py:181-215` | links= | - | - | - | _rollup_defending_aggregate, compute_defensive_credits |
| [ ] | `_defensive_credit_observation_rollup` | `tracking/defensive_credit/_orchestration.py:268-342` | links= | - | groupby@309 | - | - |
| [ ] | `_rollup_defending_aggregate` | `tracking/defensive_credit/_orchestration.py:218-265` | links= | - | groupby@251, groupby@250, groupby@249, groupby@248, set_index@235 | - | _defensive_credit_observation_rollup |
| [ ] | `compute_defensive_credits` | `tracking/defensive_credit/_orchestration.py:73-153` | links=+linker | link@96 | reindex@99-101, set_index@99-100, drop_duplicates@99 | - | - |
| [ ] | `_append_engine_companions` | `tracking/features.py:454-504` | links=+ctx | ctx@483 | - | - | _resolve_action_frame_context |
| [ ] | `_append_fixed_zone_companions` | `tracking/features.py:507-525` | links= | - | - | - | _append_engine_companions |
| [ ] | `_closing_time_per_series` | `tracking/features.py:3739-3825` | linker | link@3751 | set_index@3753 | groupby@3759 | - |
| [ ] | `_gk_influence_at_actions` | `tracking/features.py:3411-3588` | links=+linker | link@3465 | set_index@3466 | groupby@3474 | - |
| [ ] | `_make_das_transformer` | `tracking/features.py:3346-3400` | helper | - | - | - | resolve_frame_ids_by_position, _make_das_transformer.‹l›.das_features, _map_das_to_actions |
| [ ] | `_make_das_transformer.‹l›.das_features` | `tracking/features.py:3357-3395` | helper | - | - | - | resolve_frame_ids_by_position, _map_das_to_actions |
| [ ] | `_map_das_to_actions` | `tracking/features.py:3073-3133` | links= | - | - | - | resolve_frame_ids_by_position |
| [ ] | `_player_influence_at_actions` | `tracking/features.py:4469-4680` | links=+linker | link@4532 | set_index@4533 | groupby@4534 | - |
| [ ] | `_precompute_obso_lookup` | `tracking/features.py:5977-6149` | links= | - | - | drop_duplicates@6055, groupby@6046 | resolve_frame_ids_by_position |
| [ ] | `_run_values_at_actions` | `tracking/features.py:2068-2179` | links= | - | groupby@2145 | - | value_off_ball_runs |
| [ ] | `_shape_graph_at_actions` | `tracking/features.py:5710-5769` | linker | link@5729 | merge@5735-5739 | - | - |
| [ ] | `_team_shape_at_actions` | `tracking/features.py:2586-2662` | linker | link@2620 | merge@2626-2630 | - | - |
| [ ] | `actor_arc_length_pre_window` | `tracking/features.py:917-956` | helper | - | - | - | - |
| [ ] | `actor_reachable_area_m2` | `tracking/features.py:4683-4715` | helper | - | - | - | _player_influence_at_actions |
| [ ] | `actor_speed` | `tracking/features.py:357-381` | ctx | ctx@380 | - | - | _resolve_action_frame_context |
| [ ] | `add_action_context` | `tracking/features.py:529-594` | links=+ctx | ctx@570 | merge@593, set_index@592 | - | _append_engine_companions, _resolve_action_frame_context |
| [ ] | `add_actor_pre_window` | `tracking/features.py:982-1020` | links=+linker | link@1017 | merge@1019, set_index@1018 | - | - |
| [ ] | `add_cover_shadows` | `tracking/features.py:4113-4297` | links=+linker | link@4202 | merge@4290-4295, set_index@4289, set_index@4203 | groupby@4204 | - |
| [ ] | `add_das` | `tracking/features.py:3208-3343` | links= | - | - | set@3304 | _map_das_to_actions |
| [ ] | `add_defensive_credit` | `tracking/features.py:7156-7206` | links=+linker | link@7190 | merge@7205, set_index@7204, drop_duplicates@7204 | - | _aggregate_defensive_credit |
| [ ] | `add_defensive_line` | `tracking/features.py:1452-1510` | links=+linker | link@1507 | merge@1509, set_index@1508 | - | _defensive_line_at_actions, _append_fixed_zone_companions |
| [ ] | `add_ghost_gk` | `tracking/features.py:5136-5382` | links=+linker | link@5208 | isin@5317, merge@5288, set_index@5287, drop_duplicates@5284, merge@5265-5268 | merge@5335, align_join_keys@5334, merge@5275-5280, align_join_keys@5272, set@5217 | - |
| [ ] | `add_gk_influence` | `tracking/features.py:3829-3904` | links= | - | merge@3897-3902, set_index@3896 | - | _gk_influence_at_actions |
| [ ] | `add_line_break` | `tracking/features.py:1886-1938` | links= | - | - | - | detect_line_breaking, _line_break_kernel |
| [ ] | `add_obso` | `tracking/features.py:6153-6263` | links=+linker | link@6250 | merge@6256-6260, drop_duplicates@6253 | - | _precompute_obso_lookup |
| [ ] | `add_off_ball_context` | `tracking/features.py:1942-1989` | links= | - | - | - | _line_break_kernel |
| [ ] | `add_off_ball_run_values` | `tracking/features.py:2183-2269` | links=+linker | link@2262 | merge@2268, set_index@2267, drop_duplicates@2267 | - | _run_values_at_actions |
| [ ] | `add_packing` | `tracking/features.py:1621-1768` | links=+linker | link@1764 | merge@1767, set_index@1766 | - | _packing_at_actions, _append_engine_companions |
| [ ] | `add_pausa` | `tracking/features.py:6736-6828` | links= | - | - | - | add_obso |
| [ ] | `add_pitch_control` | `tracking/features.py:2852-2872` | links= | - | - | - | pitch_control_at_target |
| [ ] | `add_player_influence` | `tracking/features.py:4855-4924` | links= | - | merge@4917-4922, set_index@4916 | - | _append_fixed_zone_companions, _player_influence_at_actions |
| [ ] | `add_pre_shot_gk_angle` | `tracking/features.py:848-891` | links=+ctx | ctx@886 | - | - | _resolve_action_frame_context |
| [ ] | `add_pre_shot_gk_position` | `tracking/features.py:728-789` | links=+ctx | ctx@777 | merge@788, set_index@787 | - | _resolve_action_frame_context |
| [ ] | `add_press_commitment` | `tracking/features.py:7214-7244` | links=+linker | link@7235 | merge@7243, set_index@7242, drop_duplicates@7242 | - | compute_press_commitment |
| [ ] | `add_pressure_on_actor` | `tracking/features.py:1204-1272` | links=+linker | link@1269 | merge@1271, set_index@1270 | - | _append_engine_companions, pressure_on_actor |
| [ ] | `add_shape_graph` | `tracking/features.py:5471-5622` | links=+linker | link@5570 | merge@5621, set_index@5620, merge@5576-5580 | set_index@5560, groupby@5529, set@5512 | - |
| [ ] | `add_shot_goalmouth` | `tracking/features.py:7015-7078` | links=+linker | link@7074 | merge@7077, set_index@7076 | - | compute_shot_goalmouth |
| [ ] | `add_space_creation` | `tracking/features.py:6453-6649` | links=+linker | link@6563 | merge@6642-6646, drop_duplicates@6639 | groupby@6569 | resolve_frame_ids_by_position, add_visible_area_coverage |
| [ ] | `add_structural_pass` | `tracking/features.py:1556-1587` | links=+linker | link@1583 | merge@1586, set_index@1585 | - | _structural_pass_at_actions |
| [ ] | `add_team_shape` | `tracking/features.py:2383-2523` | links=+linker | link@2445 | merge@2521, set_index@2520, merge@2470-2474 | set_index@2439 | _append_fixed_zone_companions |
| [ ] | `back_line_high_x` | `tracking/features.py:1305-1325` | helper | - | - | - | _defensive_line_at_actions |
| [ ] | `back_n_count` | `tracking/features.py:1388-1405` | helper | - | - | - | _defensive_line_at_actions |
| [ ] | `ball_carrier_at_action` | `tracking/features.py:210-328` | links=+linker | link@285 | drop_duplicates@316, merge@288-292 | merge@308-313 | - |
| [ ] | `compactness_x` | `tracking/features.py:1328-1345` | helper | - | - | - | _defensive_line_at_actions |
| [ ] | `cover_shadow_xfns` | `tracking/features.py:4300-4461` | helper | - | - | - | resolve_frame_ids_by_position, cover_shadow_xfns.‹l›._cover_shadow_transformer, cover_shadow_xfns.‹l›._cover_shadow_transformer.‹l›._get_cs |
| [ ] | `cover_shadow_xfns.‹l›._cover_shadow_transformer` | `tracking/features.py:4348-4457` | helper | - | - | groupby@4370 | resolve_frame_ids_by_position, cover_shadow_xfns.‹l›._cover_shadow_transformer.‹l›._get_cs |
| [ ] | `cover_shadow_xfns.‹l›._cover_shadow_transformer.‹l›._get_cs` | `tracking/features.py:4372-4413` | helper | - | - | - | - |
| [ ] | `das_at_action` | `tracking/features.py:3136-3204` | links= | - | - | set@3176 | _map_das_to_actions |
| [ ] | `das_xfns` | `tracking/features.py:` | global | - | - | - | _make_das_transformer |
| [ ] | `defenders_in_triangle_to_goal` | `tracking/features.py:412-438` | ctx | ctx@437 | - | - | _resolve_action_frame_context |
| [ ] | `defensive_line_x` | `tracking/features.py:1283-1302` | helper | - | - | - | _defensive_line_at_actions |
| [ ] | `defensive_line_xfns` | `tracking/features.py:1513-1552` | helper | - | - | - | _defensive_line_at_actions, defensive_line_xfns.‹l›._defensive_line_transformer |
| [ ] | `defensive_line_xfns.‹l›._defensive_line_transformer` | `tracking/features.py:1541-1548` | helper | - | - | - | _defensive_line_at_actions |
| [ ] | `elastic_sync_xfns` | `tracking/features.py:6978-7006` | helper | - | - | - | elastic_sync_xfns.‹l›._helper |
| [ ] | `elastic_sync_xfns.‹l›._helper` | `tracking/features.py:6996-7000` | helper | - | - | - | - |
| [ ] | `ghost_gk_xfns` | `tracking/features.py:5385-5460` | helper | - | - | - | add_ghost_gk, ghost_gk_xfns.‹l›._ghost_gk_transformer |
| [ ] | `ghost_gk_xfns.‹l›._ghost_gk_transformer` | `tracking/features.py:5412-5456` | linker | link@5432 | - | - | add_ghost_gk |
| [ ] | `gk_closing_time_mean_s` | `tracking/features.py:3702-3736` | helper | - | - | - | _closing_time_per_series |
| [ ] | `gk_closing_time_min_s` | `tracking/features.py:3665-3699` | helper | - | - | - | _closing_time_per_series |
| [ ] | `gk_influence_xfns` | `tracking/features.py:3907-4104` | helper | - | - | - | resolve_frame_ids_by_position, gk_influence_xfns.‹l›._gk_influence_transformer, gk_influence_xfns.‹l›._gk_influence_transformer.‹l›._get_gi |
| [ ] | `gk_influence_xfns.‹l›._gk_influence_transformer` | `tracking/features.py:3971-4100` | helper | - | - | groupby@3994 | resolve_frame_ids_by_position, gk_influence_xfns.‹l›._gk_influence_transformer.‹l›._get_gi |
| [ ] | `gk_influence_xfns.‹l›._gk_influence_transformer.‹l›._get_gi` | `tracking/features.py:3996-4068` | helper | - | - | - | - |
| [ ] | `gk_pitch_control_share_weighted` | `tracking/features.py:3591-3625` | helper | - | - | - | _gk_influence_at_actions |
| [ ] | `gk_reachable_area_m2` | `tracking/features.py:3628-3662` | helper | - | - | - | _gk_influence_at_actions |
| [ ] | `lateral_width` | `tracking/features.py:1348-1365` | helper | - | - | - | _defensive_line_at_actions |
| [ ] | `line_breaking_ward_xfns` | `tracking/features.py:2670-2722` | helper | - | - | - | detect_line_breaking, line_breaking_ward_xfns.‹l›._line_breaking_ward_transformer |
| [ ] | `line_breaking_ward_xfns.‹l›._line_breaking_ward_transformer` | `tracking/features.py:2700-2718` | helper | - | - | - | detect_line_breaking |
| [ ] | `max_lateral_gap` | `tracking/features.py:1368-1385` | helper | - | - | - | _defensive_line_at_actions |
| [ ] | `nearest_defender_distance` | `tracking/features.py:331-354` | ctx | ctx@353 | - | - | _resolve_action_frame_context |
| [ ] | `obso_actual` | `tracking/features.py:5852-5892` | links= | - | - | - | _precompute_obso_lookup |
| [ ] | `obso_optimal` | `tracking/features.py:5936-5974` | links= | - | - | - | _precompute_obso_lookup |
| [ ] | `obso_peak` | `tracking/features.py:5895-5933` | links= | - | - | - | _precompute_obso_lookup |
| [ ] | `obso_xfns` | `tracking/features.py:6266-6340` | helper | - | - | - | obso_actual, obso_optimal, obso_peak, obso_xfns.‹l›._helper |
| [ ] | `obso_xfns.‹l›._helper` | `tracking/features.py:6317-6334` | helper | - | - | - | - |
| [ ] | `off_ball_context_xfns` | `tracking/features.py:1992-2040` | helper | - | - | - | _line_break_kernel, off_ball_context_xfns.‹l›._off_ball_context_transformer |
| [ ] | `off_ball_context_xfns.‹l›._off_ball_context_transformer` | `tracking/features.py:2014-2036` | helper | - | - | - | _line_break_kernel |
| [ ] | `off_ball_run_value_xfns` | `tracking/features.py:2272-2342` | helper | - | - | - | _run_values_at_actions, off_ball_run_value_xfns.‹l›._off_ball_run_value_transformer |
| [ ] | `off_ball_run_value_xfns.‹l›._off_ball_run_value_transformer` | `tracking/features.py:2321-2338` | helper | - | - | - | _run_values_at_actions |
| [ ] | `off_ball_xt_opponent` | `tracking/features.py:4751-4781` | helper | - | - | - | _player_influence_at_actions |
| [ ] | `off_ball_xt_team` | `tracking/features.py:4718-4748` | helper | - | - | - | _player_influence_at_actions |
| [ ] | `packing_xfns` | `tracking/features.py:1771-1827` | helper | - | - | - | _packing_at_actions, packing_xfns.‹l›._packing_transformer |
| [ ] | `packing_xfns.‹l›._packing_transformer` | `tracking/features.py:1812-1823` | helper | - | - | - | _packing_at_actions |
| [ ] | `pausa_xfns` | `tracking/features.py:6831-6901` | helper | - | - | - | add_pausa, pausa_xfns.‹l›._helper |
| [ ] | `pausa_xfns.‹l›._helper` | `tracking/features.py:6875-6895` | helper | - | - | - | add_pausa |
| [ ] | `pitch_control_at_target` | `tracking/features.py:2730-2848` | links= | - | - | groupby@2823 | resolve_frame_ids_by_position |
| [ ] | `pitch_control_xfns` | `tracking/features.py:2875-2899` | helper | - | - | - | pitch_control_at_target, pitch_control_xfns.‹l›._pc_helper |
| [ ] | `pitch_control_xfns.‹l›._pc_helper` | `tracking/features.py:2895-2896` | helper | - | - | - | pitch_control_at_target |
| [ ] | `player_influence_xfns` | `tracking/features.py:4927-5127` | helper | - | - | - | resolve_frame_ids_by_position, player_influence_xfns.‹l›._player_influence_transformer, player_influence_xfns.‹l›._player_influence_transformer.‹l›._aggregate, player_influence_xfns.‹l›._player_influence_transformer.‹l›._get_pi |
| [ ] | `player_influence_xfns.‹l›._player_influence_transformer` | `tracking/features.py:4962-5123` | helper | - | - | groupby@5002 | resolve_frame_ids_by_position, player_influence_xfns.‹l›._player_influence_transformer.‹l›._aggregate, player_influence_xfns.‹l›._player_influence_transformer.‹l›._get_pi |
| [ ] | `player_influence_xfns.‹l›._player_influence_transformer.‹l›._aggregate` | `tracking/features.py:5046-5082` | helper | - | - | - | - |
| [ ] | `player_influence_xfns.‹l›._player_influence_transformer.‹l›._get_pi` | `tracking/features.py:5005-5044` | helper | - | - | - | - |
| [ ] | `pre_shot_gk_angle_off_goal_line` | `tracking/features.py:827-844` | ctx | ctx@842 | - | - | _resolve_action_frame_context |
| [ ] | `pre_shot_gk_angle_to_shot_trajectory` | `tracking/features.py:805-824` | ctx | ctx@822 | - | - | _resolve_action_frame_context |
| [ ] | `pre_shot_gk_distance_to_goal` | `tracking/features.py:671-696` | ctx | ctx@692 | - | - | _resolve_action_frame_context |
| [ ] | `pre_shot_gk_distance_to_shot` | `tracking/features.py:699-724` | ctx | ctx@720 | - | - | _resolve_action_frame_context |
| [ ] | `pre_shot_gk_x` | `tracking/features.py:610-640` | ctx | ctx@636 | - | - | _resolve_action_frame_context |
| [ ] | `pre_shot_gk_y` | `tracking/features.py:643-668` | ctx | ctx@664 | - | - | _resolve_action_frame_context |
| [ ] | `pressure_on_actor` | `tracking/features.py:1060-1147` | links=+ctx | ctx@1107, ctx@1111, ctx@1135 | - | - | _resolve_action_frame_context |
| [ ] | `reachable_area_opponent` | `tracking/features.py:4819-4851` | helper | - | - | - | _player_influence_at_actions |
| [ ] | `reachable_area_team` | `tracking/features.py:4784-4816` | helper | - | - | - | _player_influence_at_actions |
| [ ] | `receiver_zone_density` | `tracking/features.py:384-409` | ctx | ctx@408 | - | - | _resolve_action_frame_context |
| [ ] | `shape_graph_xfns` | `tracking/features.py:5625-5707` | helper | - | - | - | _shape_graph_at_actions, shape_graph_xfns.‹l›._shape_graph_transformer |
| [ ] | `shape_graph_xfns.‹l›._shape_graph_transformer` | `tracking/features.py:5646-5703` | helper | - | - | set_index@5697, groupby@5669 | _shape_graph_at_actions |
| [ ] | `space_creation_xfns` | `tracking/features.py:6652-6727` | helper | - | - | - | add_space_creation, space_creation_xfns.‹l›._helper |
| [ ] | `space_creation_xfns.‹l›._helper` | `tracking/features.py:6699-6721` | helper | - | - | - | add_space_creation |
| [ ] | `structural_pass_xfns` | `tracking/features.py:1590-1617` | helper | - | - | - | _structural_pass_at_actions, structural_pass_xfns.‹l›._structural_pass_transformer |
| [ ] | `structural_pass_xfns.‹l›._structural_pass_transformer` | `tracking/features.py:1602-1613` | helper | - | - | - | _structural_pass_at_actions |
| [ ] | `team_shape_xfns` | `tracking/features.py:2526-2583` | helper | - | - | - | _team_shape_at_actions, team_shape_xfns.‹l›._team_shape_transformer |
| [ ] | `team_shape_xfns.‹l›._team_shape_transformer` | `tracking/features.py:2560-2579` | helper | - | - | set_index@2573 | _team_shape_at_actions |
| [ ] | `LinkReport.sync_scores` | `tracking/schema.py:311-323` | links= | - | - | - | sync_score |
| [ ] | `_resolve_action_frame_context` | `tracking/utils.py:1502-1692` | links=+linker | link@1536 | merge@1594, merge@1543 | merge@1546-1551, align_join_keys@1545 | - |
| [ ] | `add_sync_score` | `tracking/utils.py:1848-1863` | links= | - | merge@1863 | - | sync_score |
| [ ] | `sync_score` | `tracking/utils.py:1806-1844` | links= | - | groupby@1836 | - | - |
| [ ] | `validate_fov` | `tracking/utils.py:1236-1316` | links= | - | - | - | add_visible_area_coverage |
| [ ] | `xt_xfns` | `vaep/features/expected_threat.py:25-67` | helper | - | - | - | xt_xfns.‹l›._xt |
| [ ] | `xt_xfns.‹l›._xt` | `vaep/features/expected_threat.py:62-63` | helper | - | - | - | - |

**Design (r6; owner rulings R5, R7, R8, R9):**
- W9.1 **A game-aware linker.**
  - `link_actions_to_frames` builds its frame index on (game_id, period_id, frame_id) and links within each
    (game_id, period_id).
  - The game keys of `actions` and `frames` are aligned through `id_compat.align_join_keys`, so an int `game_id` on
    one side and a str on the other still link (ADR-019).
  - `link_actions_to_frames_elastic` gets the same keys. It already reads `game_id` (`utils.py:585-591`).
  - **Every game-blind helper of the linker** becomes game-aware:
    - `_count_candidates_within_tolerance` (`:1262-1268`). It is rewritten on `group_rows` without its per-action
      `iterrows` and mask, and gets an `assert_subquadratic_growth` guard (§3, ADR-073).
    - the low-coverage floor's per-period message count (`:701`, `actions["period_id"] == p`);
    - `_diagnose_time_base` (`:696`);
    - the `LinkReport` rates.
  - `LinkReport.per_period_link_rate` keeps its meaning and values for a single-game input. A new field
    `per_game_period_link_rate` carries the (game_id, period_id) key. The floor applies to the worst
    (game, period), which follows the existing "worst period, never the aggregate" rule.
  - **Single-game warning and error texts are pinned unchanged** by a test (Hyrum).
  - `on_low_coverage="raise"` can now raise on a multi-game input whose second game links badly. Today that game's
    actions silently mis-link. The CHANGELOG lists it.
- W9.2 **The pointers gain one additive column, `action_row` (int64)** (R7). It is appended last, and it holds the
  action's 0-based row position in the `actions` frame passed to the linker. There is **no** pointer `game_id`
  column. That avoids the `game_id_x`/`game_id_y` suffix break for consumers that merge on `action_id` against a frame
  that already carries `game_id` (reviewer B; `_line_breaking.py:130-146` does this today).
  - **Row order, exact key:** a stable sort on (period_id, time_seconds, game rank). The game rank is the game's
    order of first appearance in `actions`, and it only breaks ties in time. For a single-game input that is exactly
    today's `(period_id, time_seconds)` stable order. Every existing column is unchanged for a single-game input.
  - **A second additive column, `action_fingerprint` (uint64), owner ruling R9.** It is appended after `action_row`
    and binds each pointer to the identity of the action it indexes.
    - It is computed per action row as BLAKE2b with an 8-byte digest over a canonical UTF-8 record
      `"{game}|{period}|{action}|{time}"`.
    - `game`, `period`, and `action` are the `id_compat.canonical_id` strings, so `0` and `0.0` map to the same
      value; slot `action_id` is float (reviewer A). `period` goes through `canonical_id` too, so a NaN period cannot
      reach an `int()` that raises (reviewer B). `time` is `float(time_seconds).hex()` after an upcast to float64, the
      exact bits.
    - **Missing values** get the fixed token `"<NA>"`: an NA game, NA period, or NA action, and a NaN time.
    - A float32 time is upcast to float64 before `.hex()`. Upcasting is exact, so a given float32 value always gives
      the same record. A time rounded to float32 is a **different** value from its float64 original when that value
      is not exactly representable in float32, for example `12.04`, `0.1`, or `1234.56` (reviewer A). Such a time
      gives a different fingerprint, which is correct: the pointer was built on a different time. `time_seconds` is
      float64 on both sides in the package (`spadl/schema.py:15`, `tracking/schema.py:15`), so this arises only for a
      consumer that downcasts.
    - It does not depend on pandas or the platform. A committed test vector freezes the algorithm. It covers every id
      class that `id_compat` canonicalises (int, float-integral, str, numeric str, `Int64` NA, `None`), plus NaN time,
      float32 time, ±0.0 time, and a NaN period. A change to the algorithm is a contract change, so the column
      docstring states it.
    - **Cost (G2; both reviewers):**
      - The fingerprint is computed **vectorised, once per `actions` object per call**: one `canonical_id_series` pass
        per column, then one BLAKE2b loop over the joined records. The resulting array is reused by every positional
        join inside that call.
      - Nested `links=links` pass-through chains (reviewer A counted 117 sites) and the VAEP slots do not recompute it
        per hop. The shared W9.5 helper caches it for the duration of one top-level call, and never across calls.
      - **The cache is never keyed on a bare `id()`.** CPython reuses an id after the object is freed, and a call
        creates and drops temporary frames: sort copies and VAEP slots. An id-keyed cache could then serve frame A's
        fingerprints to frame B, which is exactly the silent mis-join R9 closes (reviewers A and B). Each cache entry
        holds a **strong reference** to its `actions` frame for the call. A lookup is a hit only when
        `entry.frame is actions`. The cache is cleared when the top-level call returns.
      - W0 measures its cost per match. If it exceeds 2 % of any measured step, the builder brings the measurement to
        the owner before shipping.
    - It is not named `game_id`, so it causes no suffix collision.
    - Gamestate slot copies get the same fingerprint as the row they copy, which is correct, because they are that
      action.
- W9.3 **Duplicates are legal.** VAEP gamestate slots repeat `action_id` by design: `gamestates`
  (`vaep/feature_framework.py:85-140`) shifts rows and fills the period-start rows with copies. Reviewer B measured
  `states[1].action_id = [0, 0, 1, 3, 3, 4]` (1 game, 2 periods, 6 actions). Reviewer A measured `[0.0, 0.0, 2.0, 2.0]`
  (4 actions); slot `action_id` is float. The positional join handles duplicates and multi-game input the same way,
  so nothing raises on duplicates. (The r3 author call to raise is withdrawn: it would break `compute_features` with
  any frame-aware xfn on every two-period match.)
- W9.4 **Every consumer joins positionally on `action_row`.** "Consumer" means three kinds of code, in
  `silly_kicks/` and `scripts/` (R8):
  - every function that calls either linker;
  - every function that accepts `links=`;
  - every function that reaches the pointers through a helper.

  Each consumer rule:
  - **Pointer → action** joins use `action_row`, never `action_id` alone. Existing positional surrogates
    (`features.py:5727` `_row_idx`, `_kernels.py:959`, `features.py:1104`) move to the shared `action_row`, so one
    mechanism remains (G5).
  - **Pointer → frame** joins key on (game_id, period_id, frame_id), with the action's `game_id` taken by position.
    This covers `_resolve_action_frame_context` (`utils.py:1317-1322`), which reviewer A measured mixing game 8's
    players into a game-7 action. It also covers every other pointer→frame merge on (period_id, frame_id).
  - The R2 `astype(str)` game re-keys are replaced by `align_join_keys`.
  - **Scripts (R8):** `scripts/_rq_corpus.py:60`, `scripts/tune_structural_pass_sigma.py:40`,
    `scripts/measure_cover_shadow_argmax_agreement.py:119`, and any others the census finds.
- W9.5 **Caller-supplied `links=` without `action_row`** (every pointer frame produced before this release). One
  shared helper decides, and every `links=` acceptor calls it (G5):
  - If `actions` spans one game **and** `links["action_id"]` is unique **and** `actions["action_id"]` is unique, the
    helper joins on `action_id`. That is backward compatible, and values are unchanged.
  - Otherwise it raises `ValueError`. The message says to re-link with the current linker. That covers a multi-game
    input, duplicate ids in the links (reviewer B), and duplicate ids in actions (gamestate slots). An ambiguous join
    is never guessed (AG4).
  - **A `links=` frame with `action_row` is verified at every position, on every positional join (R9).** For each
    pointer, `action_row < len(actions)` must hold. The pointer's `action_fingerprint` must equal the fingerprint of
    `actions.iloc[action_row]`, recomputed by the same function. Any mismatch raises `ValueError`, and the message
    names the first mismatched row and says "links must be built from exactly this actions frame, in this row order".
    A range check alone is not enough: a re-sorted frame of the same length passes it (reviewers A and B).
  - **Every function that re-sorts, filters, or subsets `actions` and then passes `links=` on** must remap the
    pointers through its permutation, by fingerprint, or re-link on the new frame. It must never forward the caller's
    pointers unchanged. Known sites:
    - `add_pre_shot_gk_context`: atomic, `atomic/spadl/utils.py:900` → `:990-991`; standard, `spadl/utils.py:753-754`;
    - `resolve_restart_geometry(sorted_actions, links=links)` (`spadl/utils.py:840`).

    The census of all such sites comes from the W9.6 call graph: every population member that reorders or subsets
    `actions` before a `links`-taking call. Each one is listed in the plan with line evidence.
  - The docstrings of the linker and every acceptor state the contract: "`links` must be built from exactly this
    `actions` frame, in this row order". The CHANGELOG repeats it.
- W9.6 **Registry gate (ADR-056), derived, not hand-kept.**
  - **The population** is derived from a static call graph over `silly_kicks/` and `scripts/`, iterated to a
    fixpoint:
    - the seed is every function that calls `link_actions_to_frames`, `link_actions_to_frames_elastic`, or
      `_resolve_action_frame_context`, or that has a `links` parameter;
    - then every function that calls a member is added, until nothing changes.
    - **Dispatch the static graph cannot see** (reviewer B). xfns run through lists and closures: the
      `shape_graph_xfns` closure (`features.py:5625-5660`), `lift_to_states` (`:96`, `:598-601`), and the framework's
      xfn-list calls. Two rules cover it:
      - The seed also includes every `*_xfns` factory and every xfn closure it returns, and every function that
        `lift_to_states` wraps. The call graph treats "a factory returns a closure that calls X" as an edge to X.
      - A **runtime cross-check** in the same test module patches both linkers and `_resolve_action_frame_context`
        with a recorder. The recorder identifies the caller by `(co_filename, co_firstlineno)` of the calling frame's
        code object, and maps that to the AST function node at that file and line.
        - For a decorated function, `co_firstlineno` is the first decorator's line, while the AST `lineno` is the
          `def` line (reviewer B). The map therefore uses each function node's first decorator line when it has
          decorators, and its `def` line otherwise.
        - It does not use a qualified name: Python 3.10, which is in the CI matrix (`ci.yml:56`), has no
          `co_qualname`, and generic closure names such as `_xfn` or `_shape_graph_transformer` collide (reviewer A).
        - It runs the W9.6 fixtures through every public population member and every default xfn list. It asserts
          that every observed caller maps to a population member.
        - The check is shown red with a planted, statically unreferenced xfn that calls the linker, on Python 3.10 and
          on the newest CI Python.
    - **Stop rule** (reviewer B): the fixpoint walk does not climb above the public API. Public functions of
      `silly_kicks` are members but are not expanded further, and `scripts/` entry points (`main`, CLI handlers) are
      leaves. The plan records the population size. If it exceeds 150 functions, the builder brings it to the owner
      before writing the gate.
  - The gate asserts that set exactly against a committed registry, and `_UNDERIVABLE` is asserted empty. Reviewer
    A's scan found 20 public functions reachable only through `_resolve_action_frame_context`:
    - `nearest_defender_distance`, `actor_speed`, `receiver_zone_density`, `defenders_in_triangle_to_goal`;
    - the six `pre_shot_gk_*` (`tracking/features.py:331-827`);
    - their atomic twins (`atomic/tracking/features.py:677-999`).

    They are members by derivation. The private→public coverage map is **derived** from the same call graph: each
    private member must be reachable from at least one public member, asserted.
  - **The two-game fixture** shares `period_id`, `frame_id`, and `action_id` across the two games, and offsets the
    clocks. It includes VAEP gamestate slots (duplicate ids) for every frame-aware xfn in the population.
  - **Binding fixtures (R9)**, for every `links=` acceptor, cover three cases:
    - a non-chronological `action_id` order (a mart);
    - a reversed game order ([game 8, game 7]);
    - a filtered subset.

    There are two kinds of case, with different pass conditions (reviewer B):
    - **Caller mismatch:** the caller passes pointers built on a different frame than the `actions` it passes. Green
      means the W9.5 `ValueError`. Silently wrong values are never green.
    - **Internal re-sort or filter:** the caller's `links` match the caller's `actions`, and the function itself
      re-sorts or filters (`add_pre_shot_gk_context`, `resolve_restart_geometry`, and the census). Green means
      **only** "values equal the clean call". A raise is a failure here, because these single-game calls work today
      and must keep working (Hyrum).

    **Red first:**
    - The reversed-game case is red at the branch base, because of the game-blind join.
    - The mart-order and subset cases are already green at the base, because today's `action_id` joins do not depend
      on order (reviewer B). So their red-first proof uses a **planted mutation** on the new code instead: one with
      the fingerprint verification disabled, and one that forwards the caller's pointers unchanged at a re-sort
      site. The test shows each fixture going red under each mutation. The mutations live in the test only.
  - **The game-split property** has one comparator per output type:
    - **per-action outputs** (DataFrame/Series aligned to `actions`): `f(a ∪ b)` rows equal
      `concat(f(a), f(b))` in values, realigned by position. Warnings are excluded.
    - **per-match / per-key outputs** (for example `_carrier_objective._prepare_match:62`,
      `keeper_identity._resolve_from_native:914`, `restdefense.select_rest_defense_samples:71`): the rows of game a
      in `f(a ∪ b)` equal `f(a)`, keyed by the output's own game key. The same holds for game b.
    - **the linker itself** (`(pointers, LinkReport)`): pointer rows per game equal the per-game call, with
      `action_row` offset by game a's length. `per_game_period_link_rate` equals the per-game values.
    - **pooled outputs** (calibration objectives that return one scalar or one model over all matches): exempt from
      the split property, listed in a committed exemption registry with a reason. Each one instead gets a test that
      its per-match inputs come from the per-action functions above.
  - **Red-first is per function.** At the branch base, every non-exempt population member's property test fails, or
    the registry records why it cannot fail (for example a function that never reads frame content). The
    red-at-base run is recorded in the W0 memo.
- W9.7 **Single-game bit-identity, without platform-dependent hashes** (reviewer B: CI runs ubuntu and windows,
  `ci.yml:49-55`; `test_pressure_snapshot.py:17-21` records a raw-hash CPU flake).
  - **One-time proof:** the W0 driver runs every non-exempt population member on a seeded single-game fixture and on
    the corpus matches. It runs at the branch base and at the final code, on x86 and arm64, and compares with
    `assert_bits_equal`. The base results are written per platform within the run as `.npz`/parquet, never pickle
    (ruff `S`, `pyproject.toml:257`; the pickle-free rule, AGENTS.md), and are never committed.
    The pointer comparison covers the pre-existing columns only, because `action_row` is new by design.
  - **CI guard:** the W9.6 game-split property, which compares outputs within one process on one platform, plus the
    existing test suite. No cross-platform hash is committed.
- W9.8 **Downstream.** The handback and CHANGELOG name:
  - the new `action_row` and `action_fingerprint` columns, the frozen fingerprint algorithm, and the instruction to
    join on `action_row`;
  - the binding contract: `links` must be built from exactly this `actions` frame, in this row order;
  - the W9.5 rules for legacy and for fingerprinted links;
  - the new `on_low_coverage="raise"` multi-game case.

  `docs/PRIVATE_CONSUMERS.md` is re-checked for pinned private linker helpers. The lakehouse links one match per call
  and passes `links=` (v2 `line_breaking.py:83-86`; reviewer A), so W9.5 accepts its current pointers unchanged.

**Acceptance:**
- The W9.6 gate is green, after each member was shown red at the base (or recorded as unable to fail).
- The W9.6 binding fixtures are green, after the reversed-game case was shown red at the base and the other two red
  under the planted mutations.
- The W9.6 runtime recorder cross-check is green, after the planted xfn showed it red on Python 3.10 and the newest
  CI Python.
- The `action_fingerprint` test vector is green.
- W0 records the fingerprint cost per match.
- The W9.7 one-time proof is recorded in the memo for x86 and arm64.
- The existing link tests and the pinned single-game warning texts are green.
- The two-game reproduction above gives frames [10, 30, 10, 30], with no row multiplication and no cross-game
  opponents.
- CHANGELOG and ADR entries list the value change: multi-game inputs only.

## 5. Bit-identity protocol (all speed work)

- 5.1 **Definition.** "Bit-identical" means these are all equal:
  - every float's bits, including the sign of zero;
  - dtypes, NaN/`<NA>` placement, the index, and column order;
  - the element order of returned containers.

  `np.array_equal` and `assert_frame_equal(check_exact=True)` accept `-0.0 == 0.0` (reviewer B probed this), so they
  are not enough. The comparison helper `tests/_bitwise.py::assert_bits_equal` does three things:
  1. It canonicalises NaN payloads.
  2. It compares float arrays as `uint64`/`uint32` views.
  3. It compares the rest with `assert_frame_equal(check_exact=True)` plus dtype and index checks.

  `ShapeGraph` is compared field by field, with `faces` order included. The reference is the pre-change
  implementation at the branch base.
- 5.2 **Fixtures and machines.**
  - **CI:** seeded synthetic production-shape fixtures from a generator in `tests/` with a committed seed.
    They have 22 players plus the ball, at least 2,000 frames over 2 periods, realistic gaps, NaN detections, GK
    flags, float32 storage, and mixed id dtypes.
  - **Corpus:** once on AMD x86 AVX2 (HF `cpu-upgrade`, staged per R23) through the W0 driver, on the measured
    matches. The CPU class is recorded (R22).
  - **arm64:** once on the owner's Mac Mini (R4) and once on the DGX, with the same driver and matches.
  - **One rule for every candidate that is not bit-identical by construction.** This covers non-correctly-rounded
    scalar replacements (W5.1), every new numba kernel (`_influence_rows`, W5.1, W7.2), and any BLAS/LAPACK
    replacement (W7.2). **Amended 2026-11-08 (owner ruling R13):** such a candidate ships only if the gate passes
    on **every platform we run**: AMD x86 AVX2 (HF cpu-upgrade) and arm64 (the Mac Mini and the DGX) (R22: the local PC left the rule). No other platform is in the rule.
    - **Amended 2026-11-09 (owner ruling R23):** the Mac Mini, the DGX, and HF `cpu-upgrade` run the gate on the
      uncommitted tree before the commit. The HF leg runs the staged tree (R23), so it needs no push. The earlier
      text said that an HF job can run only code that is on GitHub. That was wrong: only the commit-based
      campaign path has that rule.
    - **Every platform runs the same gate:** the candidate's parity tests, plus the W0 base-vs-final corpus compare
      on the measured matches. On HF, the base leg and the final leg are two staged W0 runs (R23), as on
      the arm64 machines: the base leg before the first package edit, the final leg after the freeze. Each leg
      checks its identity after its import, by the plan's "base, local" and "final" rules. `compare` runs
      locally on the retrieved outputs and checks both stamps (R10). The base phase uses the `BASE_SHA` public API
      only. Each HF launch needs the owner's yes, because it is billed.
    - A gate mismatch on any platform means the candidate does not ship (5.3). Any other failure (preflight,
      install, data) is not a gate result, and it goes to the owner. The revert happens before the commit (R23).
      It follows the full freeze restart rule on every platform, and it corrects every doc that describes the
      candidate.
  - A candidate that is bit-identical by construction needs the gate on the platforms available before the commit,
    not on every platform. Such a
    candidate repeats the same operations on the same inputs (W2.2 dedup, W7.1 memoisation, W7.3 pandas removal).
  - **Both backends:** each family's `FORCE_NUMPY` switch on and off.
- 5.3 **When a candidate is not bit-identical.** The builder does not loosen the gate. The builder ships the
  bit-identical variant (or the reference) and records the measured gain of the failed candidate. A tolerance-based
  speed-up comes back to the owner only as a measured proposal (max abs diff, which outputs, which consumers). The
  owner decides (AG1).
- 5.4 **Oracles.** Frozen pre-change copies of the functions under test live under `tests/oracles/`, never in the
  package. A test asserts that no `silly_kicks/` module imports them. They are deleted only by a later
  owner-approved change.

## 6. Documentation, release, and handback

- **ADR:** one new ADR (number at commit-prep). It records:
  - the public seams: `silly_kicks.backends`, the chunk parameter, the index functions, `compute_shape_graphs`, the
    DFL contract and functions;
  - the bit-identity protocol;
  - the per-action independence contract.

  It has a **separate "Approved value changes" section** for R1/R6 (the action keys), R2 (the string id keys and
  the qualifier `==`), R5 (multi-game linking), and R14 (the team-relative position labels, D-lat, D-eqf, D-tree).
  That section names each affected input class and its detection test. For R14 it records the W5.6 oracle
  relations and the W5.7 gate result. The ADR supersedes ADR-045 D5, and ADR-045 gets a pointer to it. It also records the W9.2 `action_row` and `action_fingerprint` columns (with the frozen fingerprint algorithm), the W9.3 duplicate-legality rule, and the W9.5 binding and legacy-links
  rule. The ADR amends
  ADR-008/105 (pitch-control batch), ADR-068 (frame index), and ADR-031 (DFL parse port).
- **`CHANGELOG.md`:** an entry under the cycle's `PR-Snnn`. It lists:
  - the new public API and the two new warning categories;
  - the pointer `action_row` and `action_fingerprint` columns and the `LinkReport.per_game_period_link_rate` field (W9);
  - the binding contract: `links` must be built from exactly this `actions` frame, in this row order (W9.5);
  - the new `ValueError` for legacy `links=` (no `action_row`) on multi-game or duplicate-id input, and for an
    out-of-range `action_row` or a fingerprint mismatch;
  - the new multi-game case of `on_low_coverage="raise"`;
  - the R1/R2/R5/R6 value changes, each with its input class;
  - the R14 behaviour change of `infer_positions` (every caller's lateral labels for a team that attacks +x, the
    D-tree vertical case, the −x lateral fallback for n ∉ {5, 10}, and the −x exact-y-tie case), and its new
    `ValueError` for a non-finite or zero direction. The version gets a minor bump, and the number is assigned at commit-prep.
  - the R17 fix: `compute_territorial_defense` returns an empty typed table, not `KeyError('game_id')`, when no
    action is defensive;
  - the R20 fix (scripts only): a stopped executor controller no longer leaves its workers running.
  - the R23 staged launch (scripts only): `campaign … --stage-tree` runs an uncommitted tree on HF. It refuses
    `adopt`, `verify`, and `publish`.
- **`AGENTS.md`:** within the `test_agents_md_budget.py` limits:
  - the TF-7 row (`compute_pitch_control_batch(max_frames_per_chunk=)`, `build_pitch_control_index`,
    `compute_pitch_control_batch_from_index`);
  - the TF-39 row (`compute_shape_graphs`);
  - the Providers line (the DFL functions);
  - a Key-conventions pointer for `silly_kicks.backends`.
- **Docs:** `docs/context/tracking-features.md`, `providers.md`, and `conventions-core.md` get the narrative. The
  `_spearman_combine` docstring premise is corrected (W2.5). `docs/PRIVATE_CONSUMERS.md` gets new rows where needed.
  `feature_glossary` gets the new columns. `docs/c4` gets the new public components if the C4 pin test requires them.
- **Lakehouse handback note** (to `D:\Development\_handoffs\`, not this repo). For each item it names the public
  replacement and what the lakehouse can delete:
  - `_HAS_NUMBA` (removed by #299; the guard at lakehouse `backend.py:32` must move to `silly_kicks.backends` at
    upgrade);
  - the private `_shape_graph` and `_line_breaking` imports;
  - the linker pointers' new `action_row` and `action_fingerprint` columns. Joins should be positional on `action_row`, and pointers must be built from exactly the actions frame they are passed with. Legacy pointers without it are
    accepted only for single-game unique-id input (W9.5), which is the lakehouse's current use;
  - the per-frame `compute_pitch_control` loop (v2 `pitch_control.py:76-80`). W2/W3 help v2 **only after it moves to
    the batch API**;
  - `_IDSSE_PLAYER_COLS`/`_IDSSE_TEAM_COLS` and the `calculated_frame` locator, replaced by `complete_dfl_event_bronze`;
  - v2 passes a constant `attacking_direction=+1.0` to `infer_positions`, which mislabels away teams that attack
    right-to-left. `compute_shape_graphs` uses the frames' label, so v2's labels will change on upgrade. That is a
    lakehouse defect fix and is flagged as such.
  - **Shape-graph positions (handoff 3).** v2 replaces `shape_graph_values` and its private `_shape_graph` import
    with `compute_shape_graphs(frames).players`, filtered on `shape_graph_source == "ok"`. It calls the function per
    chunk of whole frames and checks its deadline between calls (R16). The labels change for both teams (R14).
    The v1 vendored copy (`src/analytics/shape_graph_inference.py:452-458`) has the same lateral inversion, and v1
    takes the direction from team identity (ADR-051 forbids that). The note gives the W5.7 agreement numbers.

## 7. Owner rulings (2026-11-08)

- **F2. `max_frames_per_chunk` default: bounded, 2048.**
- **F3. Docstring private-import test: this cycle's modules only.** That is `tracking/pitch_control/`,
  `tracking/_shape_graph.py`, `tracking/_line_breaking.py`, `tracking/preprocess/`, `providers/sportec/`, and
  `silly_kicks.backends`.
- **R1 (replaces the r1 W6 contract). Fix the duplicate and multi-game `action_id` defect in `detect_line_breaking`**
  (W6.1, with W9). Values change only for multi-game inputs (actions concatenated across converter calls) and for
  duplicate-id inputs. Each row gets its own result, and duplicates are legal (W9.3). The detection test lands first.
- **R2 (retargets F1). Fix the ADR-019 string id keys and the cross-source raw `==` in the code this cycle rewrites:**
  - `_dispatch.py:234` `str(attacking_team_id)` dedup key → an **exact-type key**
    `(frame_key, type(attacking_team_id), attacking_team_id, decompose)`.
    - Today `str(1) == str("1")`, so the requests `(k, 1)` and `(k, "1")` merge. The second request then gets a
      surface whose `attacking_team_id` is `1`, not `"1"`. That is the detection test, and it goes red at the base.
    - A canonical key would also merge `1` and `1.0` (reviewer B), so canonical keying is not used here.
    - The exact-type key never merges requests whose returned surfaces would differ. Each surface carries its own
      request's `attacking_team_id`.
  - `features.py:5589` and `:5749` `linked["game_id"].astype(str)`;
  - the `_line_breaking.py:163-177` `str` re-key;
  - the `_resolve_idsse_team_from_qualifiers` CLU-id `==`.

  The self-referential `frames["team_id"] == tid` in `add_shape_graph`/`shape_graph_xfns` is **not** changed, and no
  `1`-vs-`"1"` merging is introduced there. Each fix gets a detection test that fails before the fix, and a CHANGELOG
  note. Values change only where the old key silently missed (the re-keys and the qualifier `==`), or over-merged
  (the `:234` `str()` key merged `1` with `"1"`). Every input that the old key handled correctly stays
  bit-identical.
- **R3. Per-row time defects in DFL events → `<NA>` + count + `DflEventTimeWarning`** (W8.1). The owner asked for
  the gold-standard option that is consistent with the other providers. The repo pattern is: a per-row data defect
  becomes a counted `<NA>` value (`ConversionReport`, ADR-051), and a whole-input contract break raises. The parser
  keeps dropping such rows, so NG4 holds.
- **R4. Scope additions: the `smooth_frames` speed-up (W7) and an arm64 gate leg on the owner's Mac Mini (5.2).**
- **R5. Fix the multi-game linker defect at the root, gold standard, after verification (W9).** The owner made the
  verification a condition. Section 4.9 records the reproduction.
- **R6. Extend R1 to `add_shape_graph`'s action path and `_shape_graph_at_actions` (W5.3).**
- **R7. The pointers gain a positional `action_row` column and no `game_id` column (W9.2).** Every consumer joins by
  position. Duplicates are legal (W9.3). This replaces the r3 author call to raise on duplicates, which both
  reviewers measured breaking VAEP gamestates.
- **R8. R5 covers in-repo `scripts/` as well as the package (W9.4, W9.6).**
- **R9. The pointers also carry an additive `action_fingerprint` (uint64)**, a frozen, platform-independent BLAKE2b
  hash of the canonical (game_id, period_id, action_id, time_seconds). Every positional join verifies it at every
  position and raises on a mismatch (W9.2, W9.5). Every function that re-sorts or filters actions remaps or re-links,
  with binding fixtures shown red first. This closes the r4 finding, raised by both reviewers, that `action_row` was
  not bound to the frame it indexes.
- **R10 (2026-11-08, plan review). The W0 base run checks package identity, not a clean tree** (W0.3). The base
  package code is proved by hash equality with `BASE_SHA`. The uncommitted driver and test files are allowed.
- **R11 (2026-11-08, plan review). `silly_kicks/_version.py` is outside the scoped code identity** (W0.4). The
  version is still assigned at commit-prep.
- **R13 (2026-11-08). The §5.2 CPU gate covers only the platforms we run:** AMD x86 AVX2 (HF cpu-upgrade) and arm64 (the Mac Mini and the DGX) (R22: the local PC left the rule). No other
  platform is in the rule. The HF leg runs after the push and before the merge (§5.2). **Superseded by R23
  (TB-PLAN-55):** the HF leg runs the staged tree before the commit.
- **R12 (2026-11-08, plan review; owner bar "no unnecessary duplication").** This cycle moves the existing copies
  onto the shared seams it uses:
  - `tests/_scale_guarded.py` `group_rows_callers` moves to the shared AST seam.
  - The three private bitwise comparators in `tests/coordination/` move to the shared bitwise helper.

  Their behaviour does not change.
- **R14 (2026-11-08). Correct `infer_positions` in place** (W5.6). The labels are in the team's own frame with
  SPADL handedness. The equal-frequency fallback and the pitch-thirds branch are fixed too. The new ADR supersedes
  ADR-045 D5. The CHANGELOG flags the behaviour change, and the version bump is assigned at commit-prep.
- **R15 (2026-11-08; amended r7.1 and r7.3 the same day). A pre-registered real-data gate plus exact mirror
  invariance** (W5.7), on all 20 public SkillCorner matches (`PUBLIC_CORPUS["skillcorner"]`; r7.3). One rule applies to each of the four cells (vertical and lateral, home and away): the agreement beats a
  role-mix-preserving permutation null (above `null_p95`), the final agreement is not below the baseline, and the
  labels show exact mirror invariance on every scored team-frame. The owner replaced the first draft's fixed 0.80
  lateral bar with the null rule, because a fixed bar was arbitrary and fell on cells that R14 cannot change
  (TB-SPEC-58). The away vertical gap is investigated and reported. A defect found there goes to the owner
  before any fix.
- **R16 (2026-11-08). Chunked calls, no stop callable** for `compute_shape_graphs` (W5.2). A test proves that
  chunks on whole-frame boundaries concatenate to one call's result.

- **R17 (2026-11-09). The empty Arm-A defect of `compute_territorial_defense` is in scope.** The W9 population
  fixture found it during plan Task 3: an action set with no defensive action (SPADL types 9, 10, 18) has an empty
  Arm-A domain, and `_assemble_samples` (`territorial_defense/_compute.py:309`) raises `KeyError('game_id')` on the
  column-less empty frame. The fix returns an empty per-defender table with the `TD_SAMPLE_COLUMNS` schema and dtypes,
  and the zero-domain report. It is a crash fix, so no existing value changes. It lands after the W0 base run (the base
  package must equal `BASE_SHA` until then). The CHANGELOG names it.

- **R18 (2026-11-09; revised in r7.5). The W0 run is restructured, so that one leg takes hours, not a day.** The
  first base leg had one unit per match and ran each step three times (plain, cProfile, `tracemalloc`). After 70
  minutes on the largest match (gradientsports 10517, 5.7 M rows) it was still in the preprocess steps, at about 6
  minutes per call. The projection was more than 10 hours per machine. The owner: "a process that takes a day cannot
  become part of our normal workflow." The rules:
  1. **Finer units.** One unit is one (provider, match, kind, step, backend), or one (provider, match, member). The
     kinds are:
     - `prereq`: the untimed prerequisites of one (phase, match, backend), computed once: the smoothed frames per
       smoothing method and the linker pointers. The output key includes the leg's `package_hash`, so a base unit
       never loads a final prerequisite (TB-PLAN-44). The other units load it.
     - `time`: one plain run of one W0.2 step on one backend: the wall clock, the peak RSS (rule 3), and the step's
       parity output.
     - `memory`: one `tracemalloc` run of `compute_pitch_control_batch` on one backend (rule 2).
     - `profile`: one cProfile run of one step on one backend: the cumulative buckets only.
     - `kernel_diff`: the W4 numba-vs-numpy kernel differences of one match.
     - `member`: one W9.7 population member on one match: the member's output only.

     A reduce stage puts the per-match files (`timings.json`, `memory.json`, `kernel_diff.json`, `census.json`,
     `parity/…`) together in sorted-key order (ADR-128). So `compare` and the memo read the same layout as before.
  2. **The W3 bound comes from `tracemalloc`, on one step only.** `tracemalloc` counts the bytes that the call
     allocates, numba arrays included (a probe saw an 80 MB `njit` array as 80.0 MB). It is the same quantity on
     every OS, so the DGX pre-freeze value (R22) sets the W3 docstring bound (rounded up), and every machine's after-run
     `memory` value is checked against it. A probe measured `tracemalloc` at 4.8 times slower on allocation-heavy
     code, so no other step runs under it.
  3. **Peak RSS per step is a planning figure, with an isolation flag** (A14-PLAN-96, TB-PLAN-39). The peak readers
     (`VmHWM`, `ru_maxrss`, `PeakWorkingSetSize`) are process high-water marks, and only Linux can reset one
     (`scripts/_stages.py` `_reset_peak_rss`). A `time` unit runs its step in a fresh child process. The child
     loads the inputs and the `prereq` output, warms up (rule 4), then reads its current RSS (the baseline) and
     its high-water mark. On Linux it resets the mark. After the step it reads the mark again.
     - The value is `isolated` when the mark was reset, or when the step raised the mark above its pre-step value.
       Then it is the step's peak above the baseline.
     - Otherwise the unit records `isolated: false`, and the pre-step mark minus the baseline as an upper bound.
     - An RSS value never sets or passes the W3 bound. The memo and the W0.5 peak statements use these values.
     - The child stamps `package_file` and `package_hash`. Its parent refuses a stamp that differs from its own
       (A15-PLAN-103).
  4. **Compilation is outside the timed and measured window** (A14-PLAN-97). Before the timed call, the unit calls
     the step once on a slice of the same match with the same dtypes: the first 200 frames of period 1 (for a
     member, the actions of those frames). That compiles every numba kernel that the step uses. The unit records
     the warm-up wall clock as `compile_s` and stamps the `SILLY_KICKS_NUMBA_CACHE` value. Both legs follow the same
     rule, and the memo reports `compile_s` per step and leg next to the timed values.
  5. **Undisturbed wall clock.** The `time` stage runs with one worker on each machine. The other stages run within
     the R19 limits. Their outputs are values, which contention does not change. The memo uses the cProfile buckets
     only as shares of a step's time.
  6. **Budget: at most 2 hours per leg on each machine (owner, 2026-11-09).** Before the first full leg, a pilot runs
     the smallest measured match (skillcorner 2015213, 1,116,466 rows) on all three machines, so no x86-to-arm64
     factor is needed (TB-PLAN-40, A14-PLAN-100). **Amended by R21:** a pilot on a new machine measures a
     sample and scales it. The projection rule is fixed before the pilot:
     - each pilot unit's wall time is scaled by the larger of the target-to-pilot rows ratio and actions ratio;
     - a `prereq` unit counts once per (phase, match, backend);
     - a parallel stage's sum is divided by its worker count, and the `time` stage is summed;
     - a leg's projection is the sum over its matches and stages.

     If any machine's projection is above 2 hours, the builder stops and brings the per-kind times to the owner.
     During a full leg, a watchdog stops the leg when its wall clock passes 2 hours, and it reports the units left.
     A watchdog stop goes to the owner, and a resume needs the owner's yes (TB-PLAN-48).
     The memo records each leg's projected and actual wall time.
  7. **The AMD gate (§5.2). Superseded by R23 (2026-11-09):** the HF leg runs the staged tree, so the
     job-private base tree below is not built. The text stays as the record. One unit is one `time`, `kernel_diff`, or `member` unit of one match. The `profile`
     and `memory` units do not run there. `ensure_base_tree` builds the job-private base tree once, **in place**
     (A14-PLAN-98, A15-PLAN-102–103, TB-PLAN-45, TB-PLAN-47):
     - `ensure_base_tree` runs on Linux only, because only the AMD gate (an HF job) uses it. It also refuses to
       run inside an `InnerPool` child, and the AMD gate stage is not `inner_parallel`, so the builder is always
       the worker process that owns its R20 group (TB-PLAN-53).
     - The locks are numbered: `dest.lock.<n>`, and the current lock is the highest number. A lock is published
       complete: its content (the builder's pid and its process group) goes into a temporary file first, and
       `os.link` publishes it. `os.link` fails when the name exists, so exactly one caller wins each number, and
       no lock is ever empty or deleted (A16-PLAN-104, TB-PLAN-50).
     - The recorded group is the builder's own R20 worker group, which exists before the lock. The builder's
       `uv sync` runs as its child in that group (TB-PLAN-51).
     - The winner of `dest.lock.1` extracts the tree into the empty `dest`, runs `uv sync --frozen` there, verifies
       it, and writes the completion marker last. There is no rename, so the editable install of the base venv
       points at `dest`. The marker is the only "ready" signal.
     - A builder that raises writes a failure record with the cause. A waiter that reads it fails its unit with
       that cause.
     - A waiter checks every 5 seconds whether the current lock's pid is alive. When it is dead, the waiter tries to
       publish the next number. Only the winner takes over. It first kills the dead builder's process group
       (`SIGKILL`) and waits until no member of that group is alive. A member counts as alive when
       `/proc/<pid>/stat` names the group and its state is not `Z`: on HF the controller is PID 1 and does not
       reap orphans, so a killed orphan stays a zombie (A17-PLAN-105, TB-PLAN-52). The wait has a limit of 60
       seconds. At the limit the winner fails its unit with the cause and the live pids, and leaves `dest` alone.
       Otherwise it clears `dest` and builds. The other waiters wait on the new lock. A second dead builder fails the unit.
     - After 30 minutes without a marker under a **live** builder, the waiter fails its unit with the cause and
       does not touch `dest`.
     - Every base-parity process (the base leg and each `probe` child) runs with `cwd=dest` and stamps
       `package_file` and `package_hash`. Its parent refuses a stamp whose import root is not `dest`, or whose hash
       is not `base_package_hash`.
     - Each base leg runs `verify_import_tree` again after its import, and writes its outputs outside `dest`. The
       guarantee is the re-verify and the stamp check: read-only mode bits do not stop root in a container
       (TB-PLAN-43).
  8. **Scope does not change:** every W0.2 step, both backends, every W9.7 member on the full corpus matches, and
     the x86 and both arm64 machines. **Amended by R21 and R22 (2026-11-09):** the members run on the R21
     slice, `profile` runs on the smallest match only, and the x86 machine is HF `cpu-upgrade`.
- **R19 (2026-11-09). W0.5 is amended for the R18 parallel stages** (A14-PLAN-99). The `time` stage keeps one unit
  at a time. The `prereq`, `memory`, `profile`, `kernel_diff`, and `member` stages may run in parallel within these
  limits:
  - **This PC** (no W0 leg since R22; 95.6 GB, 16 logical CPUs): at most 4 workers and 32 GB in total. Windows has no engine cap
    backend (`scripts/_mem_cap.py` `detect_backend` returns `"none"`), so the worker count is the limit.
  - **The Mac Mini** (24 GB, shared with the lakehouse): at most 2 workers and 12 GB in total, and only when the
    lakehouse gate is open. It runs with `--mem-backend none`, because the `rlimit` backend caps virtual memory
    (`RLIMIT_AS`), and a numba process reserves about 416 GB of it (an observed VSZ).
  - **The DGX** (119 GB, 20 CPUs): the engine's `rss` backend, with `--cap-bytes` and `--peak-rss-bytes`.
  - **HF `cpu-upgrade`** (R22, R23; 8 vCPU, 32 GB): the campaign's `peak_rss_by_stage` cap, with the worker
    count from the HF pilot (R21).

  The worker count comes from the pilot: workers × the largest unit peak of the stage, scaled to the match by the
  rule 6 ratio, must be inside the limit. The unit peak is the engine's `peak_rss_bytes`, an upper bound where its
  scope is `process`. Before each full leg, the builder states the expected peak per machine and stage to the owner.
- **R20 (2026-11-09). Orphaned executor workers are fixed in this cycle.** A W0 driver stopped with SIGTERM left its
  executor workers running with no parent: one on the DGX, and two on the Mac for about 1 h 50 m, next to a later
  leg. `_dispatch` ends its workers in a `finally` block (`scripts/hf_jobs/executor.py:525-526`), and Python does
  not run `finally` on an unhandled SIGTERM. The fix:
  - each worker starts in its own process group (POSIX `start_new_session=True`), so its `probe` children and
    inner pools are in that group (TB-PLAN-46);
  - the controller turns SIGTERM and SIGHUP into `SystemExit`, and its `finally` ends each worker's whole group;
  - each worker also ends its group and exits when its controller is gone, which covers SIGKILL. On POSIX a
    watchdog thread checks every second that the controller pid (passed as `--controller-pid`) is alive. This
    holds also under the `systemd-run --scope` wrapper, which executes in place. On Windows the controller puts
    each worker in a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, which also holds the worker's children.

  The test kills the controller (SIGTERM and SIGKILL on POSIX, `TerminateProcess` on Windows) and checks that no
  worker and no planted grandchild is alive 5 seconds later. `executor.py` is #299 code, so the move to the clean branch re-checks this
  change against #299 as merged.

- **R21 (2026-11-09). Sample and scale; the member slice; `profile` on the smallest match.** The owner: "Running
  pilots for hours makes no sense, things scale, measure a smaller percentage and multiply instead." The R18 rule 6
  projection was above 2 hours per leg (the DGX about 402 minutes serial, the Mac Mini about 439), and two thirds of
  it were the `member` units. The owner chose the slice ("Option 1"). The rules:
  1. **A pilot measures a sample.** A pilot on a new machine runs the smallest measured match on the rule 2 slice,
     for every kind. The projection scales each unit by the larger of the target-to-sample rows ratio and actions
     ratio, with the R18 rule 6 arithmetic. The full-match pilots on the DGX and the Mac Mini stand.
  2. **The member slice** (TB-PLAN-54). A `member` unit runs on a slice of its match:
     - A frame is a distinct `frame_id`. Per period, sort the period's distinct frame ids. With n of them, the
       block size is k = max(1, ceil(0.10 × n)). The slice keeps the first k and the last k frame ids, with
       all their rows. If 2k ≥ n, it keeps the whole period. That is about 20 % of each period's frames, and so
       of the match. **Corrected 2026-11-09:** r7.10 to r7.14 said 0.05, which is 10 % of the match, not the
       20 % that the owner chose; the HF pilot measured the 10 %.
     - The actions are the actions of that period whose time lies in the closed interval [first frame time,
       last frame time] of one of the two blocks.
     - The pointers do not come from the full match. Each leg links the sliced actions to the sliced frames
       with its own linker, as a `prereq` item. So no pointer names a frame outside the slice, and the member
       sees the same kind of input as on a real match. `--sample` uses the same rule, so the two modes give
       the same member inputs.
     - The selection reads only the input frames and actions, so the base and the final leg cut the same
       slice. It covers both ends of every period.
  3. **`profile` runs on the smallest match only** (skillcorner 2015213). The memo uses the cProfile buckets only as
     shares of a step's time (R18 rule 5), and a share does not need the largest match.
  4. The `prereq`, `time`, `memory`, and `kernel_diff` units keep the full matches.

  Cost if wrong: the member proof does not see a bit difference that occurs only in the middle 80 % of a period.
  The W9.6 game-split property and the seeded single-game fixture still cover every member in CI.
- **R22 (2026-11-09). The W0 machines are the DGX, the Mac Mini, and HF.** The owner: "no need to run on the
  AI-PC, Mac Mini, DGX and maybe HF are representative." The rules:
  - The W0 legs (base and final) run on the DGX and the Mac Mini (arm64), and on HF `cpu-upgrade` (AMD EPYC, x86
    AVX2), staged per R23. The local PC runs no W0 leg. It still runs the local test and lint gates.
  - The §5.2 platform rule (R13) is now AMD x86 AVX2 (HF) and arm64 (the Mac Mini and the DGX). Intel x86 leaves
    the rule. GitHub CI (ubuntu and windows) still runs the parity tests on the seeded fixtures.
  - The W3 `tracemalloc` bound comes from the DGX pre-freeze value. R18 rule 2: the quantity does not depend on
    the OS.
  - The W5.7 role gate runs on the DGX.
  - The local PC pilot data is not used. Its outputs were on the C: drive, and its timings are suspect.
- **R23 (2026-11-09). A staged HF launch runs the uncommitted tree, and the job-private base tree is dropped.**
  §5.2 said that an HF job can run only code that is on GitHub. That was wrong. The owner: "we can absolutely
  deploy things to HF without having to commit them first, that is how staging works." The pushed-and-green rule
  belongs to the commit-based campaign path (`scripts/hf_jobs/launch.py:3-5`, `preflight.check_ref`), which
  protects published artifacts. The owner approved a repo mode. The rules:
  - `python -m scripts.hf_jobs campaign <file> <command> --stage-tree`. The controller computes the working-tree
    hash T (a throwaway index, `git add -A`, `git write-tree`). It uploads `git -c core.autocrlf=false archive T`
    and its SHA-256 to the campaign bucket under `_staged/`.
  - The staged job extracts the archive into an empty `/work`. It fetches the local `HEAD` commit at depth 1 and
    points `HEAD` at it, so the git checks (for example `verify_import_tree`) work. It checks that the hash of the
    work tree equals T. Then it runs `uv sync --frozen` and the stage command, as a commit job does. The image, the
    flavor (`cpu-upgrade`), the datasets, and the in-job environment do not change.
  - The run key and the `commit` label carry `st` plus the first 10 hex characters of T, so a staged run never
    shares a store with a commit run. A staged job also has the label `staged=1`.
  - Before the first upload of a tree, the builder shows the owner the untracked paths that T carries and the
    archive size. The upload sends them to the private HF bucket, so it needs the owner's yes (TB-PLAN-59).
  - Preflight replaces `check_ref` with `check_staged`: the archive exists with the recorded SHA-256, T equals the
    current working-tree hash (a stale upload is refused), and the `HEAD` commit is on a remote branch. A green CI
    run is not required. The other checks do not change.
  - The campaign file may be uncommitted in a staged run. It is part of T, so the archive holds it, and the
    job reads the same bytes. The commit-based path keeps the committed-file rule (ADR-128).
  - `adopt`, `verify`, and `publish` refuse a staged run and write nothing. `retrieve` works, into a folder outside
    the repo. So only the commit path can make a published artifact.
  - Each staged launch needs the owner's yes, because it is billed.
  - R18 rules 5 and 6 hold on HF. A job runs the `time` stage one unit at a time (a worker limit that the stage
    declares). The controller cancels the run's jobs when the leg budget has passed since the run's first
    launch, and reports the units left. The first-launch time is recorded in the run's store, so a resume after
    a crash does not restart the clock. A resume after a watchdog stop needs the owner's yes and a new budget
    that the owner states. Each job gets a job limit (`timeout_s`) of at most the budget that is left when
    the controller launches it, so no job outlives the budget when the controller dies (TB-PLAN-56,
    A21-PLAN-115).
  - Every unit stamps its CPU (`cpu_model`, `cpu_features`). A leg whose units disagree is refused at the
    reduce. That refusal is not a gate result, and it goes to the owner (TB-PLAN-58).
  - **CPU recovery** (A20-PLAN-108). The HF base and final legs are separate jobs, so they can land on different
    CPUs, and `compare` refuses a CPU difference. The staged archives stay in the bucket until the cycle commit.
    The builder reruns the base leg from the retained base archive, as a new replica, until its CPU matches the
    final leg's. The relaunch refuses when the local campaign file differs from its copy in the archive, so
    the job arguments are the ones the tree check covers (A21-PLAN-116). Each relaunch is billed, so each needs the owner's yes, within a maximum count that the owner
    states (TB-PLAN-61). The relaunch checks the archive's SHA-256 and the in-job tree hash,
    not the current working tree, and the base stamp must still equal the `BASE_SHA` package hash. The final
    leg is never rerun from a changed tree.
  - **The parity tests count** (A20-PLAN-109). Per backend, the `parity_tests` unit requires more than zero
    passed tests, zero skipped, and zero deselected. A run where every test skips is a failure.
  - **The slice code is stamped** (A20-PLAN-113). Each leg stamps a digest of the slice code (the slice
    function, the slice link, and every driver-side function and module-level name that they reach; the package is
    left out, because the gate compares it; TB-PLAN-60). `compare` refuses when the base and final digests differ, so a slice edit
    between the legs is never read as a gate mismatch.
  - **Consequences for this cycle.** The HF W0 legs are staged runs, like the arm64 legs: the base leg now (the
    tree's package equals `BASE_SHA`, R10), and the final leg after the freeze. `compare` runs locally on the
    retrieved outputs. The candidates' parity tests run on HF as one `parity_tests` unit of the final leg: a
    child `pytest` run of their test files on both backends, which fails the unit on a non-zero exit. So the
    job-private base tree of R18 rule 7 (`materialize_base_tree`, `ensure_base_tree`, the
    `gate` phase, and the gate file) is not built. The revert path for a gate mismatch runs before the commit. The
    cycle keeps one commit, and the memo in it is complete.

- **R24 (2026-11-09). Half of the largest match, on every machine.** The largest match (gradientsports 10517,
  5.7 M rows) puts `compute_pitch_control_batch` at about 14-22 GiB on the full match, above the Mac Mini limit
  (R19: 12 GB). The owner: "can we run 1/2 GS match instead everywhere?" The rules:
  - The match runs on its first half: per period, the first ceil(n / 2) of the n sorted distinct frame ids, with
    all their rows, and the period's actions in the closed time interval of that block (`first_half`, the same
    block rule as the R21 slice). Both periods, and so both attacking directions, stay in.
  - It applies to every unit kind, on every machine (the DGX, the Mac Mini, HF), and in both legs, so the base and
    the final leg measure the same half (`--half-matches gradientsports__10517`). The R21 member slice and the
    `--sample` pilot are cut from the half.
  - The memo states that the gradientsports numbers are for half a match.
  - Each leg stamps its cut (the matches, `half_matches`, `sample`, `profile_matches`, `members`), and
    `slice_digest` covers the cut code. `compare` REFUSES two legs whose stamps differ (exit 2): a refusal is
    not a gate result, never starts the §5.3 revert, and goes to the owner (TB-PLAN-63, A23-PLAN-117-118).
  - The W3 docstring bound comes from a pre-freeze run on the WHOLE match on the DGX, which is not a W0 leg
    (A23-PLAN-120). The docstring and the memo state that basis.

  Cost if wrong: a difference that occurs only in the second half of a gradientsports period is not seen on that
  match. The other two matches stay whole.

- **R25 (2026-11-10). A W0 warm-up never fails silently.** The first base legs on `bb0c601` failed on the DGX and
  the Mac on 2 of 472 units: the gradientsports `memory` units of `compute_pitch_control_batch`. The warm-up frames
  were a concat over the first three requests. Two of them named one frame, so the index had duplicate labels,
  and the batch raised `ValueError: cannot reindex on an axis with duplicate labels`. The two `time` units of the
  same step recorded the error as `warmup_error` and passed, so their `wall_s` held the numba compile. The rules:
  - The pitch-control warm-up frames hold each distinct frame of the warm-up requests once
    (`pitch_control_warm_frames`).
  - A `profile` unit records its warm-up error as a `time` unit does. The match reduce keeps it as
    `profile_warmup_error`.
  - `export` refuses a leg in which any `time` or `profile` record has a warm-up error (`check_warmups`). A
    `memory` unit already fails on a warm-up error.

  Cost if wrong: a step whose warm-up cannot run stops the leg. The fix then goes to that step's warm-up, never to
  the gate.

- **R26 (2026-11-10). The HF W0 legs download their data.** Every read of the 232 MB gradientsports frames file
  from the `tc3` dataset mount hung in the kernel, at 4 readers, for 2332 s and 6938 s in two launches. The rules:
  - `scripts/hf_jobs/datasets.toml` gains `tc3_download`: the `tc3` repo with `access = "download"`. Its lock
    entry equals the `tc3` entry except `recorded_at`. `tc3` does not change.
  - The base and final W0 campaigns use `tc3_download`. Each shard fetches only its own files before it runs.
  - The `datasets fetch` CLI splits its argv at `--` before it parses. Its `REMAINDER` positional after `name` took
    every flag, so no download job could start. This defect is in `main` since #288; no campaign had used
    `download` before.

  Cost if wrong: each shard downloads its files (about 0.5 GB for the three matches) before it runs.

- **R27 (2026-11-10). In-job store reads go over HTTP.** Jobs write unit results through the bucket mount. In the
  base leg, a read of a result that another job wrote ran at 0.2–1.3 MB/s and then failed with
  `OSError: [Errno 5] Input/output error` (the prereq reduce, in three attempts). A read of a result that the same
  job wrote ran at about 100 MB/s. The same files downloaded over the bucket HTTP API (`download_bucket_files`) at
  29.5 MB/s, measured off HF. A one-job leg does not fit the budget: the serial `time` stage alone is about 113 min
  on HF. The owner chose to change the ADR-128 engine. The rules:
  - In an HF job, the store reads a large file over the bucket HTTP API, from the run's bucket prefix, under the
    rules below. Writes, listings, deletes, and `local_path` stay on the mount.
  - **Which copy is read** (A28-PLAN-122, TB-PLAN-67). A run rewrites or deletes these store paths in place: the
    unit result (`<base>.files/*`, `<base><suffix>`) and its `.sha256` sidecar, `.failed.json`, `.ledger.json`,
    `_summary.json`, `_inputs.json`, and `_profile.json`. A file below `HTTP_MIN_BYTES` (equal to
    `READ_LOG_BYTES`, 8 MB) is read from the mount, as before. Every record and sidecar in that list is below it.
    For a file of 8 MB or more, the store opens it on the mount and takes its `fstat` size and mtime. Then it
    gets the bucket's entry for the path (`get_bucket_paths_info`). The bucket time is the entry's `mtime`, or
    its `uploaded_at` when `mtime` is `None` (both are UTC datetimes). Both times compare as UTC epoch
    milliseconds, floored, because the bucket keeps milliseconds and the mount gives nanoseconds (TB-PLAN-75):
    - no bucket entry, or a bucket time older than the mount time: read the mount, `[read] mount newer`. This
      is a file that this job wrote and the bucket does not show yet (A29-PLAN-126, TB-PLAN-73);
    - equal sizes and a bucket time not older than the mount time: read over HTTP, `[read] http`;
    - different sizes with a bucket time not older, or no bucket time at all: read the mount,
      `[read] fallback size` or `[read] fallback no-mtime`. This is not expected. A file that is not on
    the mount is missing (`None`), as before. The checksum check (INV-3) is unchanged.
  - **Not measured:** whether the mount reports the bucket mtime for a file that another job wrote. The probe
    measures it: a cross-job read of 8 MB or more must go over HTTP.
  - **One HTTP read** (TB-PLAN-68). `http_read` in `scripts/hf_jobs/store.py` does one read and keeps no copy.
    `BucketStore.read_bytes` keeps its memo on top of it. `HttpReadStore` lives in `scripts/hf_jobs/store.py`,
    and `_stages.open_store` imports it lazily, as it imports `BucketStore` today.
  - **Every read is bounded** (A28-PLAN-123, TB-PLAN-68). `install_timeouts` bounds the metadata call. The
    transfer runs in a child process with a wall-clock limit of `HTTP_READ_FLOOR_S + size / HTTP_READ_MIN_RATE`
    (60 s plus the size at 1 MB/s). When the limit passes, the store kills the child and raises `TimeoutError`, so
    the unit fails and retries under `attempts`.
  - **Throttling** (TB-PLAN-69, TB-PLAN-74). The child writes a status record (`status.json` beside its
    destination): `ok`, `status` (the HTTP status or `null`), `retry_after` (seconds or `null`), and `error`. It
    takes `status` and `Retry-After` from an `HfHubHTTPError` response. Any other exception, a `hf_xet` transfer
    error included, records `status: null`. On 429 or 5xx, the store tries again, at most `HTTP_READ_TRIES` (5)
    times in all. It waits `retry_after` when given, else 2^n s, at most 60 s. `retry_after` is the header's
    seconds value; an HTTP-date or an unparsable value gives `null`, so the wait is 2^n s (A30 note). Any other
    failure raises.
  - **Every read says its path** (A28-PLAN-124, TB-PLAN-70, TB-PLAN-73). Each read of 8 MB or more logs exactly one
    of `[read] http`, `[read] mount newer`, `[read] fallback size`, or `[read] fallback no-mtime`, with the path,
    size, time, rate, and the reading job's launch id. The launch id (`SILLY_KICKS_LAUNCH_ID`) is a random UUID
    that each job makes when it starts, right after `set -euo pipefail` and before any other command, so every
    process of the job inherits it. A job
    whose id is empty does not start, and the probe counts an empty or missing id as another job, never as a
    match (A32-PLAN-127).
  - **The writer of a file** (TB-PLAN-78). A unit result's writer is the `launch_id` in its unit ledger. A stage's
    reduced output (`<gen>/_output*`) has no unit, so each reducer, when `SILLY_KICKS_LAUNCH_ID` is set, writes
    a marker `<gen>/_reducers/<launch_id>.json` before it writes the output. The output's writers are the set of
    those markers: two reducers of one stage write identical bytes (`reduce_stage` is idempotent), so either
    one is a writer. The id never goes into `_summary.json` or the output, because a stage without `reduce` uses
    the summary as its output, and the output must stay byte-identical for any (W, N). Off HF no marker is
    written. Any other file of 8 MB or more has no recorded writer, so a `mount newer` read of it stops the
    probe. The unit
    ledger records it beside `job_label`. `job_label` alone is not unique: a reduce launch and a one-unit run job
    share it, and so do the attempts of a relaunched shard (TB-PLAN-77). Each retry logs `[read] retry <status>`. Reads below 8 MB log nothing new.
  - **A bad environment is refused** (TB-PLAN-70). In an HF job (`SILLY_KICKS_JOB_DEADLINE_EPOCH` set), `open_store`
    refuses a store path under the mount when a store variable is missing or malformed. It never falls back to
    `LocalStore` there.
  - The job's workers and its reduce subprocess read the same way: `launch.job_env` sets the store variables, and
    every process of the job opens its store from them (TB-PLAN-72). The mount path is one constant,
    `BUCKET_MOUNT`, in `_stages.py`; `launch.py` imports it (A29 note).
  - Off HF (the DGX, the Mac, local runs, and the tests), the store is the plain `LocalStore`. Nothing changes
    there.
  - **The probe** (TB-PLAN-70, A28-PLAN-124). Before the base leg, the staged pilot campaign runs on
    `tc3_download` without `--sample` (the whole skillcorner 2015213, so its prereq outputs are about 40 MB and
    are read across jobs). It passes when:
    - the run completes, and no read raises `Errno 5`;
    - no read logs `[read] fallback` (any reason);
    - every read of 8 MB or more of a file that ANOTHER job wrote logs `[read] http`, and there is at least one.
      The writer is the `launch_id` in the unit ledger of the file's unit (TB-PLAN-76, TB-PLAN-77). There is no
      separate reduce rule: a reduce launch wrote no unit, so the own-job rule makes all its reads cross-job, and
      a reduce inside the last run job may read that job's own units as `mount newer` (A29-PLAN-126);
    - `[read] mount newer` appears only where the reading job's launch id equals the writer's. A
      `mount newer` read of a file that another job wrote stops the work;
    - the report gives the `mount newer` count per stage and per reduce.
    The report gives, per stage and per reduce, the count and total time of each tag, and the `retry` count. A
    failed condition stops the work and goes to the owner. Each billed launch needs the owner's yes.
  - **Docs** (A28-PLAN-125, TB-PLAN-71). ADR-128 is amended in place (its store section), with
    `docs/howto/compute-pipelines.md`, the `Store` docstring (`_stages.py:282`), and the `store.py` module
    docstring. The CHANGELOG entry names R25, R26 (with the #288 fetch fix), and R27, with the measurements above.

  Cost if wrong: a read of 8 MB or more costs a metadata call and a download, in a child process. Smaller reads
  stay on the mount, as before.

### Owner rulings after r7.21 (2026-11-11 to 2026-11-18)

These rulings were made in the author's session (2026-11-11, 2026-11-12) and in reviewer C's session
(2026-11-18). The ADR, the CHANGELOG, and the ledger record them too.

- **2026-11-11. The W5.7 role gate is local-only.** `scripts/validate_shape_graph_roles.py` reads the 20 public
  SkillCorner matches and their rosters through pining and runs on the DGX. It has no HF registry entry
  (`tests/scripts/test_compute_contract.py` `_LOCAL_ONLY`).
- **2026-11-11. Docstring debt for the oracle freeze.** Five public symbols of `silly_kicks/_numba_dispatch.py` stay
  without examples while the frozen-oracle closure must equal `BASE_SHA`. Ruling C5 ends this debt.
- **2026-11-12. VAEP xfn columns are float64.** `defensive_line_xfns` and `off_ball_context_xfns` emit float64 for
  every column on every input: NaN for missing, `line_break` as 1.0, 0.0, or NaN
  (`tests/tracking/test_xfn_float_columns.py`). A nullable column made the dtype depend on the data (W9.6).
- **2026-11-12. One visible-area join.** `silly_kicks/_visible_area.py` `polygons_by_action_row` serves the five
  consumers. It joins on (game_id, action_id) when `visible_area` has `game_id`. Without `game_id` it joins on
  `action_id` only when that is unambiguous, else it raises `ValueError` (the W9.5 legacy rule).
- **2026-11-12. A KickOff with no `GameSection` after a period start is a restart.** It gives no warning and no
  parser log line, and `DflPeriodReport.n_restart_kickoffs` counts it. A section-less KickOff before any period
  start stays unrecognised and warns.
- **2026-11-18. W8 stays in this cycle.** The action-time repair of the "Part Deux" cycle takes nothing from this
  branch and does not edit the event block of `providers/sportec/parse.py`, `tracking/utils.py`, or
  `tracking/_link_identity.py`.
- **2026-11-18. Reviewer C's section 8** (`D:\Development\_reviews\2026-11-17-tracking-batch-dfl-events-cycle-C.md`,
  owner: "1-5 are fine as recommended ... 6:stuck waiting for now"):
  - **C1. The record comes from the lakehouse's calls.** One before/after run on the DGX, merged tree against
    `main`, with a bit compare: the per-frame pitch-control loop against the batch call with one request per frame,
    and the per-team-frame shape-graph loop against `compute_shape_graphs`, as wall clock per frame. It replaces the
    three final W0 legs (W0.2-W0.6, section 5.2, Task 15 steps 4-6).
  - **C2. The HF leg leaves the cycle.** R23, R26, R27 and the HF part of R22 are withdrawn. The diff has no change
    under `scripts/hf_jobs/`, no `staged.py`, no `tc3_download` entry, and no W0 campaign file. The defects that the
    work found on `main` (R20, orphaned executor workers; R26, the `datasets fetch` argument) go to the owner
    session of the HF tooling as findings. The x86 proof is CI plus one run of the existing parity campaign
    (`cycle1.toml`) on the commit of this cycle, before the release (C's decision 2, which resolves TBC-GATE-04:
    every instance identical to the adopted baseline). Its launch needs the owner's yes. The owner also ruled (2026-11-18, in the author's session) that the W0
    driver `scripts/measure_tracking_compute.py` leaves with its tests, so R10, R18-R21, R24, R25, and R28 no longer
    apply. The base-leg results stay in the ledger.
  - **C3. The per-request team match goes** (TBC-PERF-03). This is the owner's yes for the W2.4 change above.
  - **C4. W9 stays as built and is closed.** A new finding goes to the owner as its own decision. The ADR table of
    alternatives gets the row "a `ValueError` for input with more than one game".
  - **C5. The oracles end in this cycle**, after the C1 compare: `tests/oracles/` and the closure digest are deleted,
    the five `_numba_dispatch` examples are added, and each batch entry point keeps an exact guard against the
    per-frame public call in the same test run.
  - **C6. Landing waits.** Superseded on 2026-11-23 (below).
- **2026-11-23. This cycle lands first** (owner, author's session: "I go first (Recommended)"). It replaces C6.
  The branch moves onto `main` at `6eee84ce` (4.144.0) with the work still uncommitted (an in-memory merge:
  5 files with entries added at the same place on both sides, all kept). The release refit of the Part Deux
  cycle merges `main` after this cycle lands; the author sends that cycle a file-by-file merge note.
- **2026-11-23. The fingerprint cost is accepted** (owner, author's session: "Accept, record it (Recommended)").
  The C1 record measured 1.92 ms per match (1,802 actions). That is 2.15 % of `smooth_frames` (0.089 s), a step that
  does not compute the fingerprint, 0.73 % of `link_actions_to_frames`, and below 0.005 % of every other step. The
  W0 rule (section 4.9, ruling R9: above 2 % of any measured step goes to the owner) went to the owner, who accepted the cost.
  The fingerprint stays the frozen BLAKE2b digest. The C1 memo in `docs/research/tracking-compute-perf/` records
  every ratio.
- **2026-11-23. Tolerance value snapshots after the oracles end** (owner, author's session: "Tolerance snapshots
  (Recommended)"). After C5, a rewritten core whose tests check only behaviour pins no output values. The author
  audits the five rewritten cores (the preprocess, the linker, the DFL parser, the shape-graph core,
  `detect_line_breaking`). Where no test pins the values, the author adds a committed snapshot from a seeded fixture,
  compared at `rtol=1e-12`, with `atol=1e-12` for values at or near zero (owner, review round 1, A-07: "rtol + atol
  1e-12 (Recommended)"). Float32-stored columns (the preprocess outputs, ADR-106) compare within one float32 ulp
  (owner, after the DGX test gate: "1 ulp for float32 (Recommended)"): on arm64 the float64 compute differs in its last
  bits, and the float32 rounding put one value in 2,132 on the neighbouring float32 in two preprocess cases, with the
  same numpy, scipy and pandas versions. The velocity and acceleration columns are finite differences of that
  float32 position, so the step reaches them amplified (up to 5.8e-5 m/s and 1.2e-3 m/s² on arm64). They compare
  within twice the worst case of one position step: 5e-4 m/s for `vx`, `vy`, `speed`, and 1e-2 m/s² for `accel_x`,
  `accel_y`, `accel` (owner: "Bound per derivative (Recommended)"). A real value change then fails on every CPU;
  last-bit differences between CPUs pass.
- **2026-11-23. Implementation review round 1** (owner, author's session; reports
  `D:\Development\_reviews\2026-11-23-tracking-batch-dfl-events-impl-{A,B}.md`).
  - **A-01, "Approve all, record (Recommended)".** Three single-game output changes are approved value changes:
    1. `sync_score` rows follow the action rows that the links were built on; the index stays `action_id`. Before,
       the rows were sorted by `action_id`. Input class: action rows that are not in `action_id` order.
    2. `add_ball_contacts` on an actions frame with a non-default index: the player-id columns are no longer `<NA>`
       (a bug fix: the old code aligned a fresh `RangeIndex` series by label).
    3. The restart-geometry tracking tiers (`resolve_restart_geometry`, `add_restart_coordinates`) read the linked
       frame on (period_id, frame_id). Input class: a provider whose `frame_id` repeats across periods.
  - **A-02, "Fix both now (Recommended)".** Defensive credit reads the linked frame on (period_id, frame_id), the
    input class of A-01 item 3. `slice_around_event` matches within (game_id, period_id) when `actions` and `frames`
    both have a `game_id` column. Each fix has a detection test that was red first; single-game output does not
    change except for the repeated-`frame_id` class.
  - **Rerun, "Fix all, one full rerun (Recommended)".** Every review fix lands in one pass. Then the tree is frozen
    again, and Task 15 runs once more: on the DGX the W9.6 game split, the W5.7 role gate and the C1 record, and
    locally the W8 DFL pass. The new freeze replaces `FREEZE_ID`, and the memos carry the new runs.
  - **A-03 (fixed under the same ruling).** `compute_shape_graphs` counts the outfield rows that have a missing
    `game_id`, `period_id`, `frame_id` or `team_id` in one `UserWarning` (AG4). This is a new warning on a new public
    function (section 8).
  - **Rerun result.** The new freeze is `c64baa66`. On the DGX, the C1 bit compare equals round 0 in every row. The
    W9.6 game split gives 267 of 267 equal. The W5.7 role gate passes with the same numbers in every cell. Locally,
    the W8 DFL pass is equal on all 7 matches. The fingerprint cost is 2.16 % of `smooth_frames` (round 0: 2.15 %),
    which the accepted ruling covers. The value snapshots pass on arm64 (15 of 15) under the tolerance rulings. The
    C1 and W5.7 memos carry the new runs.
  - **PR CI (2026-11-24).** On pandas 3, `_visible_area._keys` wrote into a read-only Copy-on-Write `to_numpy()`
    array. The fix copies the array (one line). The owner's ruling: amend the cycle commit, with no extra commit. No
    measured run executes the line, and the C1 memo lists it as the one line between `FREEZE_ID` and the commit.

## 8. Left to the builder (the plan settles these under the gates above)

- The exact internal data structures for the W2 index, the W5 core, and the W7 memoisation.
- The numba-vs-Python choice inside W5.1 and W7, gated by section 5. Any new kernel registers its own family (W1).
- Whether `_spearman_combine` stays in the package for a per-frame caller, or lives only as an oracle (AG2).
  Settled by the plan: after W2.6 it has no package caller, so it is deleted and lives only in the oracle.
- Module placement of the new public functions, behind the stated public import paths.
- Nothing else. A change to any public signature, default, column, or warning named in this document goes back to
  the owner.

## 9. Order of work

1. W0 baseline (package identity equal to the branch base, R10). It runs after the W9 population is derived and
   before the first package edit. The W9.7 base results and the W9.6 red-at-base run are also taken at the base.
2. W9 detection gate, shown red at the base, then W9 (the linker and its consumers). Later workstreams build on the
   keyed join.
3. W1.
4. W2, W3, and W4 together (pitch control).
5. W5: W5.1 first (bit-identical), then W5.6 (the label fix), then W5.2–W5.5 on the fixed labels. W5.7's
   driver is built here.
6. W6.
7. W7.
8. W8. **This ends all code edits.**
9. W0 after-run (HF AMD x86 and both arm64 machines; R22), stamped with the scoped code identity (W0.4), and the
   W5.7 role gate (the DGX).
10. Docs, ADR, and memo. These are outside the scoped identity.

W8 does not depend on the others, but the branch is created from `main` after #299 merges. So all work waits for
#299, unless the owner says otherwise.
