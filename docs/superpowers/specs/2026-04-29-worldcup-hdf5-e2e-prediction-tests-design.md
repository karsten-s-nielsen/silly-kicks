# WorldCup HDF5 fixture + e2e prediction tests in CI + DEFERRED.md cleanup

**Status:** Approved (design)
**Target release:** silly-kicks 1.9.0
**Author:** Claude Opus 4.7 (1M)
**Date:** 2026-04-29
**Predecessor:** 1.8.0 (`docs/superpowers/specs/2026-04-29-recall-based-add-possessions-validation-design.md`)

---

## 1. Problem

silly-kicks ships 5 prediction-pipeline e2e tests (VAEP fit + rate, xT fit + rate, atomic VAEP fit + rate). Each depends on `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` — an HDF5 store of pre-converted SPADL actions for the 64 WorldCup-2018 matches. The HDF5 has never been committed to the repo and is not on the user's only development machine. The session-scoped `sb_worldcup_data` fixture in `tests/conftest.py` calls `pytest.skip(...)` when the file is absent (added in 1.5.0 to avoid `FileNotFoundError`).

**Net effect:** the 5 prediction tests have been silently skipping in every CI run since silly-kicks 1.0.0 — and silently skipping locally during the user's release-ritual `pytest tests/ -v` runs — for ~9 release cycles. There has been zero actual coverage of:

- VAEP feature computation (`vaep/features.py` — 809 LOC)
- VAEP label generation (`vaep/labels.py`)
- VAEP model fitting via xgboost
- VAEP rating
- xT model fitting + rating + interpolation
- Atomic-SPADL conversion called from the atomic VAEP test
- Atomic VAEP feature computation, label generation, fit, rate

These are arguably the *core value proposition* of silly-kicks (post-conversion analytics). They've been dark for the entire 1.x lifecycle.

The diagnosis surfaced during PR-S8 brainstorming when I went looking for the WorldCup HDF5 to commit it. The file existed neither locally nor on any nearby development directory.

**Secondary problem:** `docs/DEFERRED.md` is a Phase 2-3d audit archive (early April 2026) that has been rotting — internal status inconsistency (some items marked RESOLVED in the top tables but still listed as open in lower tables), no live tracking, and out of alignment with the user's preferred convention from luxury-lakehouse (single TODO.md with explicit on-deck vs tech-debt sections). Per the "National Park Principle" (every cycle reduces tech debt slightly more than it adds), bundling a small migration into this PR is the cheap path.

## 2. Goals

1. **Restore actual e2e CI coverage on the prediction pipeline.** Vendor `spadl-WorldCup-2018.h5` so the 5 `test_predict*` tests run on every CI matrix slot.
2. **Reproducibility** — commit a build script (`scripts/build_worldcup_fixture.py`) so the fixture can be regenerated when the converter evolves (e.g., PR-S10's algorithm changes will require regen).
3. **Surface CI packaging regressions** — switch `sb_worldcup_data` from `pytest.skip` to `pytest.fail` when the file is absent, matching PR-S8's pattern for committed-fixture tests.
4. **Migrate `docs/DEFERRED.md` to TODO.md** — single source of truth aligned with luxury-lakehouse conventions; preserve audit trail in git history; remove the rotting parallel document.
5. **Lint `scripts/` in CI** — update the CI lint job to cover the new `scripts/` directory.

## 3. Non-goals

1. **No changes to the prediction tests themselves** — only the `@pytest.mark.e2e` markers are dropped. Test bodies stay as-is.
2. **No algorithmic changes** — neither to `add_possessions` (PR-S10), nor to VAEP / xthreat (no known issues, not queued anywhere).
3. **No raw event JSONs committed** alongside the HDF5. The 64 raw event files (~192 MB total) live in a gitignored cache directory `tests/datasets/statsbomb/raw/.cache/`. PR-S8's three vendored raw events (7298 / 7584 / 3754058, ~9 MB) stay as-is — they're for the boundary_metrics test, separate scope.
4. **No similar treatment for Wyscout-public** — `tests/datasets/wyscout_public/` exclusion in `.gitignore` stays. If a similar e2e need surfaces for Wyscout, that's a follow-up PR.
5. **No new dependencies** — script uses stdlib + pandas + pytables (already in `[test]` extras).
6. **No deep audit of DEFERRED.md** — only migrate explicitly-open items into TODO.md; resolved/struck-through items live in git history. Don't try to retroactively close low-impact "by design" entries.
7. **No tests for the build script itself** — it's a one-off generator. Its correctness is verified by the resulting HDF5 successfully feeding the e2e tests. Treat it like a Makefile.
8. **No ADR for the build-script convention** — this spec doc serves as the WHY record; an ADR would be over-formal for a one-off script.
9. **No fix for the audit findings being migrated** — A19 / D-9 / O-M1 / O-M6 are inventoried in TODO.md only; addressing them is future work.
10. **No touching luxury-lakehouse** — that repo is the other Claude session's responsibility.

## 4. Architecture

### 4.1 File structure

```
scripts/build_worldcup_fixture.py                                NEW   ~150-200 LOC   downloads + converts + writes HDF5
tests/datasets/statsbomb/spadl-WorldCup-2018.h5                  NEW   ~10-30 MB     vendored fixture
tests/conftest.py                                                MOD   ~+5 / -3      pytest.skip → pytest.fail
tests/vaep/test_vaep.py                                          MOD   -2 lines      drop @pytest.mark.e2e on test_predict + test_predict_with_missing_features
tests/test_xthreat.py                                            MOD   -2 lines      drop @pytest.mark.e2e on test_predict + test_predict_with_interpolation
tests/atomic/test_atomic_vaep.py                                 MOD   -1 line       drop @pytest.mark.e2e on test_predict
.gitignore                                                       MOD   +2 lines      add tests/datasets/statsbomb/raw/.cache/ exclusion
.github/workflows/ci.yml                                         MOD   +1 line       add scripts/ to lint paths
pyproject.toml                                                   MOD   +1 / -1       version 1.8.0 → 1.9.0
CHANGELOG.md                                                     MOD   +50 lines     ## [1.9.0] entry
TODO.md                                                          MOD   ~+15 / -7     remove PR-S9 line; add ## Tech Debt section with 4 migrated items
CLAUDE.md                                                        MOD   +0 / -1       remove "[docs/DEFERRED.md]" reference
docs/DEFERRED.md                                                 DEL   -127 lines    history preserved in git log
```

### 4.2 Build script (`scripts/build_worldcup_fixture.py`)

**Purpose:** download StatsBomb open-data WorldCup-2018 raw events, convert each via `silly_kicks.spadl.statsbomb.convert_to_actions`, and write a pandas HDFStore matching what `tests/conftest.py::sb_worldcup_data` expects (one `games` table + per-game `actions/game_<id>` keys).

**CLI** (argparse):

```
python scripts/build_worldcup_fixture.py [--output PATH] [--cache-dir PATH] [--no-cache] [--verbose] [--quiet]
```

| Flag | Default | Behavior |
|---|---|---|
| `--output` | `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` | HDF5 output path |
| `--cache-dir` | `tests/datasets/statsbomb/raw/.cache/` | Where downloaded raw event JSONs live |
| `--no-cache` | False | Re-download all files even if cached |
| `--verbose` | False | Per-match progress logging |
| `--quiet` | False | Errors only |

**Pipeline:**

```
1. Parse args + resolve absolute paths.
2. Fetch StatsBomb matches manifest:
   - URL: https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/43/3.json
         (competition_id=43 = FIFA World Cup, season_id=3 = 2018)
   - Cache: <cache-dir>/manifest-43-3.json
   - Cache validity: file exists AND mtime < 7 days old, unless --no-cache
   - 64 matches expected; warn loudly if count != 64
3. For each match in manifest (sorted by match_id for deterministic ordering):
   a. Download raw event JSON to <cache-dir>/events/<match_id>.json
      - Skip download if cached + not --no-cache
      - Retry up to 3 times with exponential backoff (1s, 2s, 4s)
      - Print progress: "[i/64] match_id (cache hit | downloaded {N} KB)"
   b. Adapt to silly-kicks input shape (top-level keys → typed columns):
      - game_id ← match_id
      - event_id ← e["id"]
      - period_id ← e["period"]
      - timestamp ← e["timestamp"]
      - team_id ← e["team"]["id"]
      - player_id ← e["player"]["id"] (or NaN)
      - type_name ← e["type"]["name"]
      - location ← e["location"]
      - extra ← {k: v for k, v in e.items() if k not in _top_level_keys}
      Same adapter pattern as PR-S8's e2e test.
   c. Resolve home_team_id from the per-match record in the manifest
      (match["home_team"]["home_team_id"]).
   d. Convert: actions, _report = statsbomb.convert_to_actions(
        adapted_df, home_team_id=home_team_id, preserve_native=None
      )
   e. Buffer SPADL DataFrame keyed by match_id.
   f. On conversion failure: log error, continue, mark match as skipped.
4. Build games metadata DataFrame:
   - Columns: game_id (int), home_team_id (int), away_team_id (int),
     home_score (int), away_score (int), kick_off (datetime),
     competition_id (int), season_id (int), competition_name (str),
     match_date (str)
   - One row per successfully-converted match.
   - Sorted by game_id ascending — this means games.iloc[-1] (used by
     test_predict cases as the holdout game) is deterministic.
5. Write HDFStore at output path:
   store = pd.HDFStore(output_path, mode="w", complib="zlib", complevel=9)
   store["games"] = games_df
   for game_id, actions_df in buffered_actions.items():
       store[f"actions/game_{game_id}"] = actions_df
   store.close()
6. Validate:
   - All converted match_ids present in BOTH games and actions/game_<id> keys
   - Every actions DataFrame has > 100 rows (sanity floor)
   - Final HDF5 file size < 50 MB (warn if exceeded; would suggest enabling
     stronger compression or excluding columns)
7. Print summary: total matches converted, skipped, file size, total time.
```

**Error handling:**

- **Manifest fetch fails** → fatal exit, can't proceed.
- **Individual event JSON fetch fails (after retries)** → fatal exit (deterministic fixture requires all 64).
- **Conversion of individual match fails** → log warning, skip match, continue. Validation step catches the resulting incompleteness.
- **Validation fails** → exit non-zero with detailed message.

**Idempotency:**

- Cached raw events make re-runs fast (~30 sec for adaptation + conversion + HDFStore write).
- HDF5 output is deterministic given identical inputs (pandas HDFStore writes are reproducible at the fixture level).
- Cache is keyed by `match_id` only — if StatsBomb updates a match upstream, our cache stays stale until `--no-cache` or manual deletion. Acceptable for a vendored fixture.

**Type-checking + linting:**

The script is in `scripts/`. Pyright's `[tool.pyright]` `include` is `["silly_kicks"]` so the script is not pyright-checked. Ruff reads the whole tree by default; we explicitly add `scripts/` to the CI lint job to ensure it runs there too.

### 4.3 Conftest fixture change (`tests/conftest.py:17-26`)

```python
@pytest.fixture(scope="session")
def sb_worldcup_data() -> Iterator[pd.HDFStore]:
    hdf_file = os.path.join(os.path.dirname(__file__), "datasets", "statsbomb", "spadl-WorldCup-2018.h5")
    if not os.path.exists(hdf_file):
        pytest.fail(
            f"WorldCup-2018 SPADL fixture not found at {hdf_file!r}. "
            f"This fixture is committed to the repo as of silly-kicks 1.9.0. "
            f"If absent, regenerate via: python scripts/build_worldcup_fixture.py "
            f"(or check for accidental .gitignore exclusion)."
        )
    store = pd.HDFStore(hdf_file, mode="r")
    yield store
    store.close()
```

Rationale: matches PR-S8's `pytest.fail` pattern for committed fixtures. Once the file is committed, "missing" is a packaging error worth surfacing prominently — not a silent skip that lets CI quietly regress.

### 4.4 e2e marker drops (5 tests)

| File:Line | Before | After |
|---|---|---|
| `tests/vaep/test_vaep.py:71` | `@pytest.mark.e2e` (above `def test_predict`) | line removed |
| `tests/vaep/test_vaep.py:81` | `@pytest.mark.e2e` (above `def test_predict_with_missing_features`) | line removed |
| `tests/test_xthreat.py:218` | `@pytest.mark.e2e` (above `def test_predict`) | line removed |
| `tests/test_xthreat.py:228` | `@pytest.mark.e2e` (above `def test_predict_with_interpolation`) | line removed |
| `tests/atomic/test_atomic_vaep.py:23` | `@pytest.mark.e2e` (above `def test_predict`) | line removed |

The session-scoped `vaep_model` (`tests/vaep/test_vaep.py:43`) and `xt_model` (`tests/test_xthreat.py:200`) fixtures train models. Neither has `@pytest.mark.e2e`; they're invoked transitively via the test functions that depend on them. Once the fixture file is present and tests are unmarked, they all run.

CI runtime impact: each `test_predict*` finishes in ~1-3 sec on top of the model-training fixtures (which are session-scoped — train once, reuse). Total CI overhead per matrix slot: ~5-15 seconds. Negligible.

### 4.5 `.gitignore` change

Add the cache directory exclusion:

```diff
 # Downloaded test datasets (not committed)
 tests/datasets/wyscout_public/
+
+# StatsBomb open-data raw events cache for scripts/build_worldcup_fixture.py
+tests/datasets/statsbomb/raw/.cache/
```

Note: the `tests/datasets/statsbomb/` exclusion was removed in PR-S8 to allow the 3 boundary-metrics fixtures. We keep that change and add the more-specific `.cache/` subpath exclusion.

### 4.6 CI lint-path expansion (`.github/workflows/ci.yml:20`)

Current:
```yaml
      - run: ruff check silly_kicks/ tests/
      - run: ruff format --check silly_kicks/ tests/
```

After:
```yaml
      - run: ruff check silly_kicks/ tests/ scripts/
      - run: ruff format --check silly_kicks/ tests/ scripts/
```

Pyright include stays `["silly_kicks"]` (per `pyproject.toml`) — build scripts aren't worth full type-checking.

## 5. DEFERRED.md → TODO.md migration

### 5.1 What lives in DEFERRED.md today

127 lines, frozen 2026-04-06. Contains audit findings from Phase 2-3d:

- ~30 RESOLVED items (struck-through or marked RESOLVED) — Phase 3b/3c/3d work that closed them
- ~10 still-open items distributed across architecture / security / optimization / documentation audits
- Some internal status inconsistency (e.g., O-2 / O-2b / O-2c marked RESOLVED at the top under "Phase 3c: Converter Rewrite" but still appearing in the optimization audit table further down — these are RESOLVED per CHANGELOG 0.1.0)

### 5.2 What migrates to TODO.md

Cross-checked each open item against the actual code state and current TODO.md tracking:

| Audit ID | Disposition | Action |
|---|---|---|
| A9 | Already in TODO.md | No-op |
| A15 | "By design, low ROI" — accept as design choice | Note in CHANGELOG; not in TODO.md |
| A16 | "YAGNI — 4 converters" — accept as design choice | Note in CHANGELOG; not in TODO.md |
| A17 | "Diminishing returns" — partial refactor done | Note in CHANGELOG; not in TODO.md |
| A19 | Low impact, never revisited | Migrate to TODO.md tech debt |
| S5 | "Standard for libraries" — accept as design choice | Note in CHANGELOG; not in TODO.md |
| D-8 | Already in TODO.md | No-op |
| D-9 | Minor cleanup, not done | Migrate to TODO.md tech debt |
| O-M1 | Minor optimization | Migrate to TODO.md tech debt |
| O-M6 | Minor optimization | Migrate to TODO.md tech debt |

All RESOLVED items live in `git log -- docs/DEFERRED.md` after deletion — audit trail preserved without working-tree clutter.

### 5.3 New `## Tech Debt` section in TODO.md

```markdown
## Tech Debt

| # | Sev | Item | Context |
|---|-----|------|---------|
| A19 | Low | Default hyperparameters scattered across 3 learner functions | Extracted to named constants in `learners.py`; could centralize further but low impact. Audit-source: DEFERRED.md (Phase 2 architecture audit) |
| D-9 | Low | 5 xthreat module-level functions (`scoring_prob`, `get_move_actions`, etc.) not underscore-prefixed but not re-exported | Implementation helpers technically public API. Audit-source: DEFERRED.md |
| O-M1 | Low | Full `events.copy()` at top of StatsBomb `convert_to_actions` (`spadl/statsbomb.py:78`) | Defensive copy — could shrink on demand. Audit-source: DEFERRED.md |
| O-M6 | Low | Temporary n×3 DataFrame for StatsBomb fidelity version check (`spadl/statsbomb.py:171`) | Audit-source: DEFERRED.md |
```

### 5.4 CLAUDE.md change

```diff
- See [TODO.md](TODO.md) for tracked work. Audit history in [docs/DEFERRED.md](docs/DEFERRED.md).
+ See [TODO.md](TODO.md) for tracked work.
```

### 5.5 DEFERRED.md deletion

`git rm docs/DEFERRED.md` in the same commit. History preserved.

## 6. Test plan + TDD ordering

(For a build-script PR, the script itself doesn't get unit tests — its correctness is verified by the e2e tests it produces data for. Test ordering reflects "build it → wire it up → verify end-to-end".)

1. **Write `scripts/build_worldcup_fixture.py`.** Implement all pipeline stages.
2. **Run the script:** `python scripts/build_worldcup_fixture.py --verbose`. Verify HDF5 created at `tests/datasets/statsbomb/spadl-WorldCup-2018.h5`, file size 10-30 MB, validation summary clean.
3. **Update `tests/conftest.py`** — `pytest.skip` → `pytest.fail` change.
4. **Drop `@pytest.mark.e2e`** on the 5 tests (single line removal each).
5. **Run full pytest suite:** `uv run pytest tests/ -v --tb=short`. Expected:
   - All previously-passing tests still pass.
   - 5 newly-running `test_predict*` tests pass.
   - Zero remaining skips for the prediction-pipeline tests (other unrelated skips may exist for other reasons — flag if so).
6. **Update `.gitignore`** (cache directory exclusion).
7. **Update `.github/workflows/ci.yml`** (add `scripts/` to lint paths).
8. **Update `pyproject.toml`** (version bump 1.8.0 → 1.9.0).
9. **DEFERRED.md migration:**
   - Add `## Tech Debt` section to TODO.md with 4 migrated items.
   - Remove PR-S9 line from `## Open PRs` in TODO.md.
   - Update CLAUDE.md to remove the DEFERRED.md reference.
   - `git rm docs/DEFERRED.md`.
10. **Add CHANGELOG entry** (`## [1.9.0]`).

## 7. Verification gates

```bash
# Match exact CI pin to avoid local/CI drift (per feedback_ci_cross_version memory)
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113

# Lint + format (CI lint job equivalent — note new scripts/ path)
uv run ruff check silly_kicks/ tests/ scripts/
uv run ruff format --check silly_kicks/ tests/ scripts/

# Type-check (silly_kicks/ only — pyright include is unchanged)
uv run pyright silly_kicks/

# Full pytest suite (all e2e markers on test_predict cases dropped)
uv run pytest tests/ -v --tb=short
```

**Expected baseline output:** prior baseline (PR-S8 post-merge) was 555 passed, 9 skipped. Of the 9 skipped, 5 are the prediction-pipeline `test_predict*` cases that PR-S9 unblocks. After PR-S9: ~560 passed, ~4 skipped (any remaining skips are unrelated non-e2e conditional skips — flag if a different number surfaces). Pyright + ruff zero errors.

`/final-review` runs after all gates pass — mandatory before commit per `feedback_commit_policy`.

## 8. Commit cycle

Per `feedback_commit_policy` and `feedback_no_commits_without_explicit_approval`: literally one commit per branch, single-commit ritual with explicit user approval at each gate.

Hook scope (narrowed during PR-S8): only `git commit` and destructive ops are sentinel-gated. Routine push, PR create / merge / comment, and tag pushes proceed on chat approval alone.

```
1. All gates green + /final-review pass
2. User approves → git add + git commit (one commit, branch feat/worldcup-hdf5-e2e-prediction-tests)
3. User approves → git push -u origin feat/...
4. User approves → gh pr create
5. CI green → user approves → gh pr merge --admin --squash --delete-branch
6. User approves → git tag v1.9.0 + git push origin v1.9.0  # auto-fires PyPI publish workflow
```

## 9. Out of scope (queued follow-ups)

### PR-S10 — `add_possessions` algorithmic precision improvement

Unchanged from PR-S8's deferral. The 64-match WorldCup HDF5 from PR-S9 is now available for parameter sweeping in PR-S10 (vs PR-S8's 3-match set), making default-tuning measurements much more reliable. See `project_followup_prs.md`.

### Tech-debt items in TODO.md

A19 / D-9 / O-M1 / O-M6 are inventoried but not addressed here. They're low-priority polish; pick them up as inline cleanups during related future work (the National Park Principle).

## 10. Risks + mitigations

| Risk | Mitigation |
|---|---|
| HDF5 size exceeds 50 MB → repo bloat warning territory | Build script validates size; pyproject default `complib="zlib" complevel=9` keeps it tight. If oversize, switch to blosc compression or downsample columns. |
| StatsBomb open-data manifest URL drift | URL stable since 2018, low risk. If breaks, build script fails fast at fetch step with clear error. |
| Match download flakiness during initial build | Retry-with-backoff in script handles transient failures. Persistent failure → exit non-zero, surfaces clearly. |
| Conversion fails on a specific match (data quirk) | Build script logs + skips the match; final validation flags incompleteness. Manual investigation if a specific match consistently fails. |
| HDF5 format reproducibility across pandas/pytables versions | pandas HDFStore writes are deterministic at the fixture level. Local 3.14 vs CI 3.10/3.11/3.12 might produce byte-different but logically-equivalent files. Mitigation: tests check semantics (column sets, lengths), not file checksums. |
| Pre-existing prediction tests fail when actually run (latent bugs since 1.0.0) | This is exactly what we want to discover! If failures emerge, surface them clearly — they're real regressions in the prediction pipeline that have been silently masked. Address inline if small (1-2 hours) or defer to a focused follow-up PR. |
| StatsBomb open-data license drift | Same license as PR-S8's vendored fixtures — non-commercial redistribution permitted under same terms. Build-script downloads and the resulting derivative HDF5 are both covered. License attribution lives in `tests/datasets/statsbomb/README.md` (added in PR-S8). |
| `pytest.fail` annoys local devs who deliberately delete the file | Clear remediation in the message points at the build script — `python scripts/build_worldcup_fixture.py`. One command to recover. |

## 11. Acceptance criteria

1. `scripts/build_worldcup_fixture.py` exists and runs cleanly (`python scripts/build_worldcup_fixture.py --verbose` produces the HDF5 in 5-10 min cold cache, ~30 sec warm).
2. `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` is committed; file size in 10-30 MB range.
3. `tests/conftest.py::sb_worldcup_data` calls `pytest.fail(...)` on missing fixture (was `pytest.skip`).
4. The 5 `test_predict*` cases have no `@pytest.mark.e2e` marker.
5. `uv run pytest tests/ -v` shows the 5 prediction tests passing (no skips on the prediction pipeline).
6. `tests/datasets/statsbomb/raw/.cache/` is in `.gitignore` (excluded from commits).
7. `.github/workflows/ci.yml` runs `ruff check` and `ruff format --check` on `scripts/` in addition to `silly_kicks/` and `tests/`.
8. `pyproject.toml` version is `1.9.0`. Tag `v1.9.0` is pushed → PyPI auto-publish workflow succeeds.
9. `docs/DEFERRED.md` is deleted; `TODO.md` has a `## Tech Debt` section with 4 migrated items (A19 / D-9 / O-M1 / O-M6); `TODO.md`'s `## Open PRs` no longer lists PR-S9; `CLAUDE.md` no longer references `docs/DEFERRED.md`.
10. CI matrix (lint + ubuntu-3.10/3.11/3.12 + windows-3.12) all pass on the PR branch before merge.
11. `/final-review` passes before commit.
