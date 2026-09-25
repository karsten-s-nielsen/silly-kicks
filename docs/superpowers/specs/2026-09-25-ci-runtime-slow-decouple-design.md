# CI runtime: regenerate durations + decouple the slow suite — design

**Status:** draft, pending independent `/review-spec`.
**Type:** architectural (changes CI job topology + rewires two anti-rot guards).
**Scope:** `.github/workflows/ci.yml`, `tests/test_ci_shard_wiring.py`, `tests/test_ci_slow_gating_wired.py`, committed `.test_durations`. Docs/infra only — **no package change, no version bump, no PyPI release** (mirrors the AGENTS.md restructure cycle).
**ADR:** ADR-074 (pytest-split sharding), ADR-023 (slow-gating). No new ADR — this refines both; the topology change (slow moves from an inline primary-leg selection to a dedicated job) is a **delivery-mechanism** change, so the amendment note fits **ADR-023** at least as well as ADR-074 — place it on ADR-023 (with a back-reference from ADR-074's sharding note).

## 1. Problem

CI wall-clock regressed to ~20 min against the ~10 min target, with the three shards of the binding leg misaligned. Two stacked causes, measured (not inferred):

- **Stale `.test_durations`.** Last measured 2026-08-27 (4.96.0, commit `2194d02`). The current suite collects 9772 `not e2e` tests; the stale file had 8126 entries — **~17% unmeasured**. pytest-split falls back to count-mode for unmeasured tests, which clumped one primary-leg shard to ~19.3 min while its siblings sat at ~11.4/11.0.
- **The slow suite runs inline on one leg.** `@pytest.mark.slow` selects **249 tests = 13.6 min = 37% of the 36.5-min `not e2e` runtime**, and ADR-023 runs them on the primary leg (ubuntu-3.12) *inside* that leg's three shards. So the primary leg is structurally heavier than every other leg by the slow total — this is the "misalignment between the parts" that regeneration alone cannot remove.

## 2. Goal

Binding job wall-clock **under ~10 min**, a **symmetric** shard matrix (every leg runs the same marker selection, so no leg is structurally heavier), and the runtime coverage proof (`shard-reconcile`) **preserved and extended** — never weakened. No test may silently run in zero shards or zero jobs.

## 3. Measured baseline (fresh CI-measured durations)

Regenerated `.test_durations` on the binding ubuntu-3.12 leg, unsharded, `not e2e` (incl slow): **9823 entries** (superset of the 9772 collected; the delta is parametrization drift), 36.5 min total, max single test 130.3 s. Previously-unmeasured heavy families now present (positioning 93, match_outcome 88, win_probability 63, agents_md_budget 10). Slow subset: **249 tests / 13.6 min**, all 249 matched in the durations file. Non-slow: 22.9 min.

Per-job fixed overhead, measured from run 36170683006 step timings:

| Leg | checkout+setup+install+doctest+2×collect |
|---|---|
| ubuntu-3.12 | ~75 s (~1.25 min); install 39 s |
| windows-3.12 | ~138 s (~2.3 min); install 84 s (binding) |

Balanced test-phase (greedy LPT on the fresh durations) and projected **job wall-clock = overhead + test-phase**, binding leg:

| Option | Change | Binding job wall | New jobs |
|---|---|---|---|
| **A only** (regen durations) | remove clumping | ubuntu-3.12 primary (incl slow) 12.2 test + 1.25 = **~13.4** | 0 |
| **B1** (`--splits 3→4`) | more shards | ubuntu-3.12 primary 9.1 + 1.25 = **~10.4** | +4 |
| **C** (decouple slow) | symmetric matrix + dedicated slow job | windows non-slow 7.6 + 2.3 = **~9.9**; slow job 6.8 + 1.25 = **~8.1** | +2 net |

**A + C** is chosen. A alone lands ~13.4 (still over). B1 is a one-number bump but throws 33% more runners at a balanced-but-heavy suite, lands ~10.4 (over), and leaves the structural asymmetry in place. C attacks the root cause (isolate the 37% invariant heavy tail), lands under 10, uses fewer runner-minutes than B1, and honors ADR-023's own intent (slow = "expensive AND interpreter-invariant → run once") — a dedicated job is the natural home. B1 is recorded as the cheap fallback if C is rejected.

## 4. Design

### 4.1 Part A — regenerate durations (already applied, uncommitted/unstaged, in the working tree)

A temporary `durations-capture` job (committed on this branch as `5bc80ed`, PR #256) ran the full `not e2e` selection unsharded on ubuntu-3.12 with `--store-durations` and uploaded `.test_durations` as a CI-measured artifact. The regenerated file replaces the committed one, and the capture job is removed in the same change (net-zero `ci.yml` vs `main` for that job; verified `git diff --quiet origin/main -- ci.yml`). Local capture mis-balances (CI ~2× local, per-interpreter relative timings differ), so the file **must** be CI-measured — this is why the throwaway job exists.

The one file serves **both** split spaces: pytest-split filters to the `-m` selection and balances on whatever durations it has for the selected tests. The fresh file covers the slow tests (249/249 matched), so both the non-slow matrix split and the slow-job split balance on real numbers.

### 4.2 Part C — decouple the slow suite

**Matrix `test` job — symmetric.** Every leg runs one bulk step:
```
pytest tests/ -m "not e2e and not slow" --splits 3 --group ${{ matrix.shard }} -p no:randomly --benchmark-skip --tb=short --durations=25
```
The `matrix.primary` include block and the two complementary `if: matrix.primary` / `if: !matrix.primary` bulk steps are removed — there is now exactly one bulk step, unconditional. The node-ID `--co` steps (for reconcile) lose their `matrix.primary && 'not e2e' || ...` ternary and use the constant `not e2e and not slow`. Matrix axes otherwise unchanged: `os ∈ {ubuntu, windows}`, `python ∈ {3.10, 3.11, 3.12}`, `shard ∈ {1,2,3}`, minus windows-3.10 / windows-3.11. Doctest (shard-1-per-leg), pandas-major recording (shard-1-per-leg), the numba cache step, and `SILLY_KICKS_ASSERT_INVARIANTS`/`NUMBA_CACHE_DIR` env are all retained verbatim.

**New `slow` job.** Runs the invariant heavy tail once, sharded ×2 on the primary interpreter only (ADR-023: slow is platform/interpreter-invariant, so one leg suffices):
```yaml
slow:
  runs-on: ubuntu-latest
  strategy:
    fail-fast: false
    matrix:
      slow-shard: [1, 2]
  env:
    SILLY_KICKS_ASSERT_INVARIANTS: "1"
    NUMBA_CACHE_DIR: ${{ github.workspace }}/.numba_cache
  steps:
    - checkout (pinned SHA, as test job)
    - setup-python 3.12, cache: pip
    - actions/cache numba  # SAME key pattern as the test job: numba-ubuntu-latest-3.12-${{ hashFiles(<the two njit patterns>) }}
    - pip install -e ".[kloppy,xgboost,das,test]"
    - run: pytest tests/ -m "slow and not e2e" --splits 2 --group ${{ matrix.slow-shard }} -p no:randomly --benchmark-skip --tb=short --durations=25
    # reconcile artifacts:
    - shell: bash; pytest -m "slow and not e2e" --splits 2 --group ${{ matrix.slow-shard }} -p no:randomly --co -q | grep '^tests/.*::' | sort > slow-shard-nodeids.txt  → upload slow-shard-nodeids-<slow-shard>
    - if slow-shard == 1: pytest -m "slow and not e2e" -p no:randomly --co -q ... > slow-full-nodeids.txt  → upload slow-full-nodeids
    - if slow-shard == 1: pytest -m "not e2e" -p no:randomly --co -q ... > combined-full-nodeids.txt  → upload combined-full-nodeids   # ubuntu-3.12 `not e2e` incl slow, for the §4.3 cross-check
```

**`shard-reconcile` — extended, `needs: [test, slow]`.** Three proofs, all fail-loud on zero artifacts (the pandas-span idiom):
1. **Matrix legs** (existing, unchanged mechanism): per leg, the 3 shard node-ID sets partition that leg's full `not e2e and not slow` set — union == full, pairwise-disjoint. All legs now share the same marker selection.
2. **Slow job** (new): the 2 slow-shard node-ID sets partition the `slow and not e2e` full set — union == slow-full, pairwise-disjoint.
3. **No-drop cross-check** (new, load-bearing — see §4.3): on the primary interpreter (ubuntu-3.12), `non-slow-full(ubuntu-3.12) ⊎ slow-full == combined-full` (the `not e2e` incl-slow collection), disjoint union. This is the proof that decoupling dropped nothing — that the slow tests removed from the matrix reappear exactly once in the slow job.

`EXPECTED_N` stays 3 for the matrix partition; the slow partition uses its own N=2.

### 4.3 Coverage-proof invariant (why §4.3 cross-check is mandatory)

A green CI proves nothing failed, not that a test *ran* (the recurring trap in this repo). Splitting the suite into two marker selections opens a silent-drop hole: a test that is neither collected by `not slow` nor by `slow` (e.g. an `e2e`-adjacent marker interaction, or a future third marker) would vanish and CI would stay green. The `non-slow ⊎ slow == not-e2e` cross-check on the primary interpreter closes it by conservation — the same enumerate-and-conserve discipline the existing per-leg reconcile uses.

**This invariant is the one part of the design that lives ONLY in CI runtime** — the reconcile job body + the slow job's artifact uploads — and is exercised by no local test, so its only verification would otherwise be "green CI", which §4.3 itself says proves nothing ran. A future `ci.yml` edit that drops a slow-job upload or a reconcile proof would leave every guard green and silently reopen the hole. It is therefore pinned structurally by a new anti-rot guard (§4.4, `test_ci_slow_reconcile_wired`), the same `ci.yml`-reading idiom as `test_ci_pandas_span_wired` / `test_ci_shard_wiring`.

### 4.4 Guards rewired (TDD — assertions first)

The two guard tests currently assert the *present* structure and would need to change to expect C. TDD order: edit the assertions to expect the C topology (they go **red** against the current `ci.yml`), confirm red, then edit `ci.yml` until green.

**`tests/test_ci_slow_gating_wired.py`:**
- `test_ci_bulk_steps_partition_with_slow_gating` → rewrite as `test_slow_is_a_dedicated_job_not_inline`: assert (a) the `test` job has **exactly one** bulk (`pytest tests/ ... --benchmark-skip`) step and it is **unconditional** (no `matrix.primary` `if`); (b) that step's marker is `not e2e and not slow`; (c) a top-level `slow` job exists whose bulk step marker is `slow and not e2e`; (d) the `slow` job runs on ubuntu / python 3.12; (e) no `matrix.primary` remains in the `test` matrix include.
- `test_slow_marker_set_is_non_empty` → unchanged.

**`tests/test_ci_shard_wiring.py`:**
- `_sharded_cmds()` → generalize to collect sharded commands from **both** `jobs.test` and `jobs.slow`.
- `test_shard_axis_is_contiguous_1_to_N` → assert `test` matrix `shard == [1,2,3]` (unchanged) **and** `slow` matrix `slow-shard == [1,2]` contiguous.
- `test_splits_value_matches_shard_count` → `test` job `--splits 3` with `--group ${{ matrix.shard }}`; `slow` job `--splits 2` with `--group ${{ matrix.slow-shard }}`.
- `test_every_sharded_command_pins_collection_order` → extend to the slow command.
- `test_shard_reconcile_job_exists_and_needs_test` → assert `shard-reconcile` `needs` includes both `test` and `slow`.
- `test_numba_cache_key_covers_all_njit_files` → assert the njit-file coverage holds for the cache step in **both** `jobs.test` and `jobs.slow` (the slow job compiles the same kernels; a missing key silently no-ops Lever C there too).

**`tests/test_ci_slow_reconcile_wired.py` (NEW — closes CI-SPEC-01):** a `ci.yml`-reading anti-rot guard (pandas-span idiom) for the §4.3 conservation invariant, which lives only in CI runtime. It asserts topology **structurally on the parsed YAML** (jobs, steps, `needs`, upload/download `name`/`pattern`, `--co` markers), **plus exactly ONE text-grep** of the reconcile step's `run` scalar for a required sentinel comment (the non-vacuity anchor, below) — the only text check, everything else structural:
- The `slow` job has **three** `upload-artifact` steps: the per-shard set named `slow-shard-nodeids-<slow-shard>` (guard matches by the `slow-shard-nodeids` prefix), plus the two single artifacts `slow-full-nodeids` and `combined-full-nodeids`; the corresponding `--co` collection steps use markers `slow and not e2e` (sharded + full) and `not e2e` (combined) respectively.
- `shard-reconcile` `needs` includes `slow`, and it **downloads** all three (the sharded set by glob `slow-shard-nodeids-*`; the two single artifacts `slow-full-nodeids` and `combined-full-nodeids` by their exact name as a literal pattern — a `-*` suffix would NOT match a suffix-less single artifact) — downloading them is the structural proof that proofs #2/#3 have their inputs; a reconcile that dropped the conservation check would have no reason to fetch `combined-full`.
- The reconcile step body references the conservation identity (a required sentinel: the guard greps the step `run` for the distinctive tokens tying `slow`+`combined` together, e.g. a `# CONSERVATION:` marker comment the reconcile body must carry) — a deliberately load-bearing comment so the AST/text check is non-vacuous, mirroring the `test_numba_cache_key` non-vacuity assertion.
- Non-vacuity: the guard itself asserts it found ≥1 of each, so an empty/renamed job fails rather than passing on zero matches.

**`--co` de-ternary (CONSIDER, folded here):** the matrix `--co` node-ID steps drop the `matrix.primary && 'not e2e' || 'not e2e and not slow'` ternary for the constant `not e2e and not slow`; `test_ci_slow_gating_wired` asserts no `matrix.primary` token remains anywhere in the `test` job (bulk and `--co` steps alike).

No new shuffle-plugin, doctest, pandas-span, lint-pin, or publish-guard wiring changes.

### 4.5 Downstream doc-consistency (surfaced in final-review)

The topology change makes several existing docs describe a mechanism that no longer exists (`matrix.primary` / "runs inline on the primary leg"). These are corrected as part of the change — a false statement in the always-loaded rules file is not acceptable:

- **`AGENTS.md`** (Testing bullet): the `@pytest.mark.slow` fragment drops the retired `primary: true` and states "run once in a dedicated `slow` job", adding `test_ci_slow_reconcile_wired`. Stays a Testing bullet (the 600-char Key-conventions cap does not apply); file stays well under the 27750 B ceiling.
- **`docs/context/ci.md`**: the slow-gating paragraph is rewritten to the dedicated-job mechanism + the conservation cross-check; the legacy "ubuntu primary leg" phrase in the durations-regen note becomes "ubuntu-3.12 leg".
- **`tests/fixtures/agents_md_invariant_inventory.json`** (the frozen AGENTS.md-restructure oracle): the `slow-gating` entry's `required_tokens` swaps the now-retired `primary: true` for **`@pytest.mark.slow`**. This is **not laundering and not a shrink**: `@pytest.mark.slow` is equally present in the pinned `CLAUDE.md`@77286f4 snapshot (so `test_inventory_tokens_exist_in_pinned_source` stays honest), the entry keeps two tokens (no count reduction, `INVARIANT_COUNT` unchanged), and the slow-gating invariant remains pinned in `AGENTS.md` by `test_ci_slow_gating_wired`. The retired token is removed **with its referent** (the deleted `matrix.primary`) — the dead-parameter discipline — rather than forcing `AGENTS.md` to carry a string that is now false.
- **ADR-023 / ADR-074**: the delivery-mechanism amendment note (§ ADR line).

## 5. What stays unchanged

Doctest (`--doctest-modules`, shard-1-per-leg), `pandas-span` job + `test_ci_pandas_span_wired`, `benchmark` standalone job, numba `actions/cache` (Lever C), `SILLY_KICKS_ASSERT_INVARIANTS`, lint pins, publish guard, `.test_durations` as the single balance source. The slow-marker *set itself* is untouched — no test gains or loses `@pytest.mark.slow`.

## 6. Out of scope / sequenced

- **Staleness guard (F).** The regression recurred because nothing catches durations rot (silent from 4.96.0). A guard (CI warns/fails when collected-vs-measured coverage drops below a threshold) is the durable recurrence fix, orthogonal to A/C. **Deferral is human-approved, verbatim (2026-09-25):** "We will do C as well and fold it into the same feature/cycle with minimal commits … When C is ready, we can consider folding in a staleness guard as well." So F is sequenced to be reconsidered once C lands — a recorded human decision, not an author-initiated deferral. It is not built in this cycle.
- **B1 (`--splits 3→4`).** Rejected in favor of C (§3), retained as the cheap fallback.
- **No version bump / PyPI.** Docs + CI-infra only.

## 7. Risks & mitigations

- **Coverage silently weakened.** Mitigated by §4.3's conservation cross-check + the existing per-leg reconcile; guard-assertions-first TDD.
- **Slow-split imbalance.** The regenerated durations cover all 249 slow tests → both shards balance on real numbers (verified). If a future slow test is unmeasured, count-mode fallback applies to it alone — same graceful degradation as the matrix.
- **Runner cost.** +2 jobs (slow ×2) but fewer total runner-minutes than B1 (slow runs on 2 jobs, not smeared across the primary leg's 3 shards; every leg's per-shard work drops by removing slow).
- **Windows still binding on install (2.3 min overhead).** Out of scope for this cycle; the pip cache + numba cache already target it (Lever C).

## 8. Testing / verification

- Guard tests red-first against current `ci.yml`, then green after the `ci.yml` edit (evidence: paste the red run then the green run).
- Full local suite green (`pytest -m "not e2e"`), lint at CI scope (`ruff check silly_kicks/ tests/ scripts/` + `--format --check`), `pyright` bare.
- The runtime proof (matrix partition, slow partition, no-drop cross-check) can only be observed on a real CI run — verified on the PR's CI, not locally.
- Independent `/review-impl` on a frozen tree before any merge.

## 9. Commits (minimal, squashed at merge)

One feature branch (`ci/regen-test-durations`), squash-merge. Logical commits: (1) A — regenerate `.test_durations`, remove the temporary `durations-capture` job; (2) C — `ci.yml` topology (symmetric matrix + `slow` job + extended reconcile), the two guard rewrites (`test_ci_slow_gating_wired`, `test_ci_shard_wiring`), and the new anti-rot guard `test_ci_slow_reconcile_wired`. The spec + plan docs commit with C (docs travel with the change). Squash at merge collapses to a single clean commit. Each commit and the merge are separately owner-gated.
