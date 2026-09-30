# DAS corpus-parity: pandas-2 reference subprocess (Option C)

**Status:** Draft for review.
**Date:** 2026-09-27.
**Branch:** `feat/combined-provenance-dgx` (PR #261; additional commits on top of `3ca609f`, coordinated with the F1b/T10 session; no version bump / no PyPI publish this cycle).
**Supersedes (in part):** the divergence-exclusion approach of `docs/superpowers/specs/2026-09-26-das-native-design.md` §6.8 / §7.2 (the D-OFF / D-KEY parity-exclusion classes — see "Amendments to the native-DAS spec" below).
**Root-cause record:** `D:\Development\_handoffs\das-parity-attribution-divergence-ROOTCAUSE.md`.

## 1. Problem

The native-DAS corpus-parity driver (`scripts/validate_das_native_parity.py`, spec §7.2) runs FOUR legs
over the owner-tier corpus and compares the native engine against `accessible-space==2.0.15`. Run on the
real corpus it shows a ~1–2 % frame tail where native and the reference diverge (team-grain up to
|Δ| ≈ 17). The prep commit `3ca609f` attributed this to divergence classes (D-OFF / D-KEY) and added a
`divergences` block that holds those frames out of the headline parity.

That diagnosis was wrong. **The native engine is correct.** The divergence is a reference-side
ENVIRONMENT artifact: `accessible-space==2.0.15` is a pandas-2-era library, and the parity driver runs it
under **pandas 3**. Under pandas 3 Copy-on-Write, the array `accessible-space` builds internally
(`PLAYER_POS = dfp.values.reshape(F, P, C)`) is **read-only**. Its offside pre-processing does the one
in-place mutation `PLAYER_POS[PLAYER_IS_OFFSIDE, :] = np.nan`; on any 150-frame chunk that contains an
offside player this raises `ValueError: assignment destination is read-only`, which the library CATCHES
and silently skips offside — keeping offside attackers the native engine correctly removes.

Proof (DGX, `~/das-parity/venv`, pandas 3.0.6, accessible-space 2.0.15), exemplar frame
`(1886347, 1, 19584)`: with the read-only array the reference team DAS = 27.536 (offside skipped,
`core.py:195` warning fires); forcing a writeable copy makes it 10.156164 — EXACTLY the native value —
and the warning disappears. The offside step is the only in-place mutation of a transform-derived array;
CoW cannot be disabled in pandas 3.0 (the option is deprecated and inert). The golden oracle matches at
1e-12 because it was frozen under pandas 2 (writeable arrays).

A second, independent `accessible-space` defect surfaced while root-causing: the DAS entry points
(`get_dangerous_accessible_space` / `get_individual_dangerous_accessible_space`) pivot on `frame_id`
ALONE (`transform_into_arrays`), so frames that reuse a `frame_id` across periods COLLIDE on a whole-match
call. This is what D-KEY (`is_d_key = n_periods > 1`) was blanket-excluding.

## 2. Goals / Non-goals

**Goals**
- Make the §7.2 parity compare the native engine against a FAITHFUL `accessible-space` reference — the
  library behaving as designed, i.e. under pandas 2, with offside respected and frames keyed
  collision-free.
- Make the silent-offside-skip class (and any reference-environment regression) UNREPRESENTABLE: the
  parity run fails loudly rather than silently comparing native against a broken oracle.
- Remove the divergence-exclusion machinery that was chasing this artifact; replace it with the minimal,
  correct reason-based accounting.
- Preserve the offline reduce-path test (the stubbed reference leg) and the golden gates unchanged.
- Record, as documented consequences, that `accessible-space==2.0.15` is silently wrong under pandas 3
  (CoW offside; `frame_id`-only keying) and that the native engine is immune — independent evidence for
  the ADR-107/108 migration and the non-obvious reason the parity leg shells out to a pandas-2 env.

**Non-goals**
- No change to the native / numba / periodic legs, the engine, the golden fixtures, or any runtime code.
- No version bump, no PyPI publish, no CHANGELOG entry this cycle (a CHANGELOG line lands at release).
- No speculative "residual non-offside divergence" machinery: Option C makes the reference faithful, so
  there is nothing left to exclude; the fail-loud guard covers regressions instead.
- No auto-provisioning of the reference environment inside the driver (a network install inside a
  provenance-stamped, clean-tree-gated run is disqualified — ADR-037/052).

## 3. Design

### 3.1 Architecture

The reference leg stops calling `accessible-space` in-process (pandas 3, broken). Instead it marshals the
match's scored frames to a temporary parquet, invokes a **pandas-2 Python interpreter** on a **sk-free**
reference script, and reads the team/player AS+DAS back from parquet. The native, numba and periodic legs
are unchanged and still run in the production pandas-3 process. The reference now runs in the golden
oracle's own regime: Python 3.12 + pandas 2.x + accessible-space 2.0.15.

The subprocess is spawned once per match (inside `_measure_match`, matching the `for_each` per-match
shard model — resumable, one temp parquet per match). The pandas-2 cold import is measured at ~0.44 s on
`py2ref` (DGX); across ~980 matches that is ~8 min of import overhead on a ~14 CPU-hr run (~1 %), so a
persistent reference worker is NOT worth the complexity (YAGNI) — per-match spawn stays.

### 3.2 Single-sourced, sk-free reference module

New module `scripts/_das_reference_leg.py`. Top-level imports are `pandas` and `numpy` ONLY — no
`silly_kicks`, and `accessible_space` is imported LAZILY inside the functions that call it. This lets the
driver and CI import the module for the constants / gate / keying helper without `accessible-space`
present (CI has neither), while the pandas-2 subprocess imports the library only when it runs the leg:

- `_X_OFFSET`, `_Y_OFFSET`, `_REFERENCE_COMMON`, `_reference_lib_frames(frames)` — moved verbatim from
  the driver (the byte-for-byte golden-generator recipe). NOTE: the driver's current `_REFERENCE_COMMON`
  comment claims "`test_das_native_parity_driver` gate-checks this against the generator", but **no such
  gate test exists today** (verified). This design ADDS that gate (§3.7) — a NEW test asserting
  `scripts/_das_reference_leg._REFERENCE_COMMON` equals `_generate.py::_COMMON` — and fixes the stale
  comment to point at it. Single-sourcing the recipe in this module keeps the new gate meaningful.
- `reference_leg_arrays(frames) -> dict` — the current `_reference_leg` body (prepare lib frames, call
  the two `accessible-space` entry points, read `ReturnValueDAS` arrays, dedupe team per frame, key
  player rows), PLUS collision-free frame keying (§3.3) and the fail-loud guards (§3.5).
- `__main__(in_parquet, out_parquet)` — read frames, run `reference_leg_arrays`, write two parquet tables
  (team: `game_id, period_id, frame_id, as, das`; player: `game_id, period_id, frame_id, player_id,
  as, das`) plus a one-row `env` table (`pandas`, `numpy`, `accessible_space`, `python` version strings).

The driver imports `_REFERENCE_COMMON` / `_reference_lib_frames` from this module (pandas 3) only for the
gate and any in-process use; it never calls `accessible_space` itself.

### 3.3 Collision-free frame keying (makes the D-KEY delete safe)

Inside `reference_leg_arrays`, before the `accessible-space` call, replace the `frame_id` fed to the
library with a synthetic globally-unique key = the dense rank over `(game_id, period_id, frame_id)`
(mirroring `accessible-space`'s own `unique_frame_col = np.arange(...)` idiom in `get_das_gained`). Map
the returned arrays back to the real `(game, period, frame)` via the driver's existing row-order
bookkeeping. DAS is a per-frame-independent computation, so this is value-neutral except that it stops
`transform_into_arrays` merging frames that share a `frame_id` across periods. Multi-period matches
therefore rejoin the headline parity, and the D-KEY class is eliminated at the root.

### 3.4 Reference-environment locate (your (i): documented prerequisite, fail-loud)

- The reference interpreter is provisioned once by the owner (already built at `~/das-parity/py2ref`):
  `python3.12 -m venv py2ref && py2ref/bin/pip install "accessible-space==2.0.15" "pandas<3"`.
- The driver resolves it from `--reference-python` (CLI) or `SK_DAS_REFERENCE_PYTHON` (env). If the path
  is absent or not executable, `SystemExit` with a message that includes the exact provisioning command.
- Before the corpus pass, the driver invokes the reference interpreter ONCE and checks
  `int(pandas.__version__.split(".")[0]) < 3` and
  `importlib.metadata.version("accessible-space") == "2.0.15"`, raising (explicit `if not …: raise`,
  never a bare `assert` — `-O` strips asserts) on mismatch. **`accessible_space.__version__` does not
  exist** (verified on `py2ref`: `hasattr(accessible_space, "__version__")` is `False`); the version
  MUST come from `importlib.metadata.version("accessible-space")` (verified → `"2.0.15"`). (This composes
  with the subprocess-side guards in §3.5 — defense in depth.)
- CI and the reduce-path test never spawn the subprocess (they inject the stub, §3.7), so no CI change.

### 3.5 Fail-loud guard (your (b): make the bad outcome unrepresentable)

Inside `reference_leg_arrays` (runs in the pandas-2 subprocess). All three are explicit `if not …: raise`
(never a bare `assert` — `-O` strips asserts):

- `warnings.filterwarnings("error", message="Offside not properly detectable")` — the silent-skip
  warning becomes a `ValueError` that aborts the run. If offside is ever silently disabled again, the
  parity fails instead of comparing against a broken oracle.
- raise unless `int(pandas.__version__.split(".")[0]) < 3` — the environment `accessible-space==2.0.15`
  requires.
- raise unless `importlib.metadata.version("accessible-space") == "2.0.15"` — pin the oracle; catch a
  silent env drift that would change the reference. (NOT `accessible_space.__version__` — that attribute
  does not exist.)

### 3.6 Reduce / accounting (your (a): surgical delete + minimal correct accounting)

**Delete** (they exist only to feed the exclusion this design removes):
- `_d_off_per_frame`, the `is_d_off` shard column and its plumbing in `_frame_flags`, `_measure_match`
  (the `match_row["is_d_off"]` / cast), `_team_rows`, `_player_rows`.
- `is_d_key` (the `n_periods > 1` derivation), `_mark_divergences`, `_divergence_block`, `_divergent`,
  and the `"divergences"` key in `reduce_parity`.

**Keep and re-anchor to `reason == Reason.OK`:**
- `_frame_flags` is retained but emits `(game_id, period_id, frame_id, reason)` only (drop `is_d_off`);
  the `reason` column stays on team/player rows so the reduce can filter. (A non-OK frame yields
  native NaN vs a fictional reference value — a finite-mask mismatch — so the clean set must be the
  OK frames.)
- `reduce_parity` computes the headline `team`/`player` grade and the `finite_mask_mismatches` over the
  `reason == Reason.OK` rows (the §7.2 "finite-mask mismatches must be 0 outside the divergence classes"
  contract). The finite-mask over OK frames must read 0.
- The existing `reason_counts` (from the match rows) IS the per-class §7.2 divergence/degrade accounting:
  `reason_ball_nan` (D-BALLNAN), `reason_poss_team_absent` (D-POSSABSENT) and the other NaN-degrade
  reasons are the real documented classes, already counted. No separate `divergences` block is needed.
  NOTE the count SOURCE: `reason_counts` is summed from the per-match `reason_*` columns on the **match**
  rows (`_measure_match` writes them from `packed.reason`), NOT from the per-row `reason` on team/player
  rows. The per-row `reason` column feeds only the clean filter. (This distinction matters for the
  re-anchored test below: it must set the count on the match row.)
- Retain the nullable-dtype handling the deleted `_mark_divergences` carried: on a grain-mixed
  `pd.concat` the `reason` column arrives object-typed with NaN, so the clean filter casts
  `reason.astype("Int64").fillna(int(Reason.OK)).astype("int64")` before comparing to `Reason.OK`
  (a plain object `.fillna` downcast is deprecated in pandas 3).

`_SHARD_SCHEMA_VERSION` bumps `das-native-parity-2` → `das-native-parity-3` (the `is_d_off` column is
removed and the clean-set semantics change).

### 3.7 Test seam + existing-test disposition

The `reference_leg=` injection point in `run_corpus` / `_measure_match` stays. The reduce-path test keeps
injecting the frozen golden reference outputs — it never spawns the subprocess, needs no
`accessible-space` and no network, and now exercises the reason-based clean filter.

**Disposition of the 6 tests in `tests/scripts/test_das_native_parity_driver.py` the delete would touch**
(these go red unless handled; the delete must leave the suite green):

- `test_full_reduce_path_is_schema_complete_and_counts_reconcile` (:110–121) — **EDIT**: drop the
  `divergences`-block schema assertions (:110–121); keep the team/player/quad/timings/`finite_mask`/
  `reason_counts` schema checks (all still valid).
- `test_native_reproduces_the_reference_leg_within_parity` (:138–140) — **EDIT**: drop the
  `divergences.d_off_frames==0` / `d_key_frames==0` / `excluded_rows` assertions; keep the parity
  `< 1e-6`, `finite_mask_mismatches == {0,0}`, and finite `quadrature_shift`. (S01/S05 are single-period
  full-team scenes → all `reason == OK` → the headline is the whole corpus.)
- `test_d_off_per_frame_flags_frames_with_fewer_than_two_defenders` (:282) — **DELETE** (its target
  `_d_off_per_frame` is deleted). Also delete the `_off_frame` helper (:223), used only by this test.
- `test_reduce_excludes_d_off_rows_from_headline_and_counts_them` (:305) — **DELETE**: it asserts a
  `reason == OK` frame with a large gap is EXCLUDED via D-OFF; under this design such a frame is
  intentionally in the headline, so the behavior is gone.
- `test_reduce_counts_and_excludes_d_key_frames_colliding_across_periods` (:345) — **DELETE**: D-KEY
  exclusion is replaced by collision-free keying in the reference module. Its intent is re-homed to the
  NEW keying unit test below (the reduce no longer handles collisions).
- `test_reduce_headline_finite_mask_excludes_degrade_reason_frames` (:396) — **RE-ANCHOR**: keep the
  `finite_mask_mismatches["team"] == 0` assertion (the non-OK `BALL_NAN` row is excluded by the
  `reason == OK` filter — unchanged behavior); replace the `divergences.d_ballnan_frames == 1` assertion
  with `reason_counts["reason_ball_nan"] == 1`, and set that count on the **match** row (per §3.6 the
  count source is the match row, not the team row).
- The section header comment (:218–220 "Divergence exclusion (spec 6.8/7.2)") is updated/removed with the
  deleted tests.

**New unit tests (no `accessible-space`, no network):**
- `scripts/_das_reference_leg._REFERENCE_COMMON` == `_generate.py::_COMMON` gate (the NEW gate, §3.2).
- The sk-free module imports WITHOUT `accessible_space` present — import the module (its
  `_REFERENCE_COMMON` / `_reference_lib_frames` / keying helper) in an environment where
  `import accessible_space` would fail, and assert it succeeds (guards the lazy-import contract of §3.2,
  today only transitively enforced).
- Collision-free keying: a synthetic two-period frame set with a reused `frame_id` maps back to distinct
  `(game, period, frame)` rows (the unique-frame-rank helper in isolation) — re-homes the D-KEY intent.
- The fail-loud guards raise: the offside warning-as-error, `pandas` major ≥ 3, and
  `importlib.metadata.version("accessible-space") != "2.0.15"` (monkeypatched), each aborting.
- The reduce drops the `divergences` block, keeps `reason_counts`, and grades over `reason == OK`
  (synthetic shards with an injected non-OK frame → excluded from the headline, counted in
  `reason_counts` via the match row).

Golden gates (`test_das_engine_parity`, the golden fixtures) are untouched — native is unchanged.

### 3.8 Provenance

The artifact records the reference environment returned by the subprocess (`pandas`, `numpy`, `python`
versions, and the accessible-space version via `importlib.metadata.version("accessible-space")` — not the
non-existent `__version__` attribute) alongside the existing `run_platform` / `run_machine`, and folds
those version strings into `declare_inputs` (`input_contract`) so the oracle's environment is part of the
provenance digest. The `das-reference` dev extra is updated to declare the reference-env contract
(`accessible-space==2.0.15`, `pandas<3`) as documentation of the pin; the `py2ref` interpreter itself is
provisioned manually (§3.4), not via the extra (the extra would pull `silly_kicks` into the reference
venv, which must stay sk-free).

## 4. Data flow

```
scored frames (pandas 3, arrow-backed)
  -> driver writes temp parquet (dtype-stable) in a system tempdir (tempfile.mkdtemp) -- NEVER inside the
     repo tree (require_clean_tree, ADR-037, would trip; shard dir stays clean); cleaned up per match
    -> py2ref/bin/python scripts/_das_reference_leg.py <in.parquet> <out_dir>
         [pandas 2] fail-loud guards -> _reference_lib_frames -> unique-frame key
                    -> get_dangerous_accessible_space + get_individual_...  -> team/player/env parquet
    <- driver reads team/player parquet -> dict{team_keys, team_as, team_das, player_keys, player_as, player_das, env}
  -> _team_rows / _player_rows (joined on (game,period,frame[,player]))
  -> shard (reason column, no is_d_off)
-> reduce_parity: grade + finite-mask over reason==OK; reason_counts; timings; NO divergences block
-> metrics.json (+ reference env in provenance)
```

## 5. Error handling (fail-loud, layered)

1. Reference-python missing/not executable → `SystemExit` with the provisioning command (§3.4).
2. Pre-pass version probe: pandas major ≥ 3 or accessible-space ≠ 2.0.15 → abort (§3.4).
3. Subprocess startup asserts (pandas < 3, accessible-space == 2.0.15) → non-zero exit → driver raises
   (§3.5).
4. Offside warning-as-error inside the subprocess → non-zero exit → driver raises (§3.5).
5. Subprocess non-zero exit or missing output parquet → the driver raises (never a silently-empty
   reference leg that would read as a vacuous parity pass).
6. Temp parquet lives in a system tempdir OUTSIDE the repo tree and is removed per match, so the
   clean-tree guard (ADR-037) and the shard directory are never polluted (SPEC-04).

## 6. Amendments to the native-DAS spec (`2026-09-26-das-native-design.md`)

- §6.8 / §7.2: D-OFF and D-KEY are removed as parity-EXCLUSION classes. D-KEY is eliminated at the root by
  the collision-free keying (§3.3). D-OFF (`< 2` finite defenders) is corpus-vacuous (real matches always
  have ≥ 2 defenders — it matches zero corpus frames); it is not specially excluded, and if such a frame
  ever appeared it would surface through the `reason == OK` finite-mask contract rather than be hidden.
- The §7.2 divergence accounting is the `reason_counts` NaN-degrade classes (D-BALLNAN, D-POSSABSENT, …)
  plus the requirement that the finite-mask mismatch over OK frames is 0.

## 7. ADR / docs (this cycle; recorded, not measured)

- Amend ADR-107 / ADR-108 Consequences: `accessible-space==2.0.15` silently disables offside under
  pandas-3 CoW (read-only `PLAYER_POS`, caught `ValueError`) and conflates frames that reuse `frame_id`
  across periods (`frame_id`-only pivot); the native engine is immune to both. This is why the §7.2
  parity leg shells out to a pinned pandas-2 environment — a future maintainer must not "simplify" the
  subprocess away and silently restore the broken comparison.
- Add a `docs/context/` note (tracking-features or a DAS-parity note) capturing the same, so the rule is
  discoverable when the code is touched.
- CHANGELOG line deferred to release (owner: no version/publish this cycle).

## 8. Rollout / branch coordination

- Work lands as additional commits on `feat/combined-provenance-dgx` (PR #261), coordinated with the
  F1b/T10 session which owns the provenance/artifact cycle. Merge preserves commits; no version bump, no
  PyPI publish this cycle.
- Owner-run validation: after the fix, re-run the corpus parity on the DGX with
  `SK_DAS_REFERENCE_PYTHON=~/das-parity/py2ref/bin/python` and confirm the headline parity returns to
  ~1e-12 with a 0 finite-mask over OK frames and the `reason_counts` accounting intact. The smoke MUST
  include at least one **multi-period match with a reused `frame_id` across periods** — the case the
  collision-free keying (§3.3) fixes and the reduce no longer excludes — to confirm those frames land in
  the headline and match (the accessible-space `frame_id`-only pivot is an external-lib behavior verified
  only indirectly; this smoke exercises it end-to-end).
- No commit / push / tag without explicit per-commit owner approval.

## 9. Risks

- **Marshalling dtype drift** (pandas 3 → parquet → pandas 2): mitigated by parquet's typed schema and by
  `_reference_lib_frames` re-coercing string/id columns inside the subprocess; covered by an owner-run
  smoke on one match before the full corpus.
- **Two sessions on one branch**: rebase/push coordinated with the owner; the reduce-path test and native
  legs are independent of the F1b/T10 work.
- **Reference-env drift over time**: pinned (`accessible-space==2.0.15`, `pandas<3`) and asserted at
  three layers (§5); stamped into provenance.

## 10. Out of scope

Repurposed residual-divergence machinery; sk install in the reference venv; auto-provisioning; any engine
/ golden / native-leg change; version bump / publish / CHANGELOG this cycle.
