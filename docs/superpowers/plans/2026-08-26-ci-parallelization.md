# CI Parallelization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring silly-kicks CI wall-clock from ~22 min to **< 10 min** by duration-sharding the test suite across parallel jobs (`pytest-split`), caching numba's compiled kernels, and preserving every CI-integrity contract — with two anti-silent-drop guards.

**Architecture:** Replace the monolithic per-leg `test` job with an `os × python × shard[1..N]` matrix; each job runs one duration-balanced `pytest-split` group on its own runner (no `xdist` memory contention). Benchmark becomes a standalone parallel job. A static wiring guard + a runtime node-ID-set reconcile job prove the shards partition each leg's suite.

**Tech Stack:** GitHub Actions, `pytest-split` (dev/test-only), `actions/cache` (numba + pip), the existing `pyyaml`-parsing `test_ci_*_wired` guard idiom.

## Global Constraints

- **SLA is the deliverable: a real CI run < 10 min wall-clock (the SLOWEST leg), measured via `gh run view` — never `total/N` arithmetic.** (spec §1, §8)
- **Zero coverage loss.** Every OS×Python leg runs its full marker selection; `@slow` stays primary-only (ADR-023); benchmark/doctest/pandas-span all still run. Proven by the node-ID reconcile job. (spec §6, §8)
- **N=3 is the owner-set working hypothesis, VALIDATED by the measured floor** (longest single test 39 s → ubuntu shard budget ~6.8 min has vast headroom; Windows is install-bound ~8:15). N lives in TWO literals (`shard: [1,2,3]` + `--splits 3`) kept consistent by the static guard; changing N is a two-line edit. (spec §2, §4)
- **Windows is the binding leg** (undivided 1:49 install); numba + pip caching are prioritized there. (spec §2 N2)
- **GitHub Free = 20 concurrent jobs, ACCOUNT-WIDE.** N=3 ⇒ ~14 peak-concurrent. Do not exceed. (spec §2)
- **No new runtime dependency** (`pytest-split` in `[test]` only); **no paid runners**; **no coverage-reducing test rewrites** (spec §2 Lever B / N3).
- **Collection order is load-bearing:** every sharded command carries `-p no:randomly`; a static guard asserts no collection-shuffling plugin is active. (spec §4, §6a / B3)
- **Commit structure (owner-set):** commit 2 = sharding infra (`ci.yml`, guards, `pytest-split`, `.test_durations`, caching) — **`.test_durations` in the SAME commit**; a 3rd commit only if Lever B produces test edits. No squash. **NO `git commit` without explicit owner approval.** (spec §9)

---

## Task 0: Record the measured floor and confirm N

**Files:**
- Reference only (no code): the `--durations` profile from the brainstorm.

**Interfaces:**
- Produces: the documented floor that justifies N=3, cited in the PR body / CHANGELOG.

- [ ] **Step 1: Record the profile in the PR/CHANGELOG prose.** Full primary selection
  (`pytest -m "not e2e" --durations=30`, local py3.14, 7930 tests / 19:27): longest single test
  **39.12 s** (`test_train_script_smoke_position_only`); top cluster = xcross train-smokes (39/38 s),
  calibration cache-equivalence (37 s), then DAS/gkdv tests 9–32 s. **No multi-minute indivisible test.**
- [ ] **Step 2: State the derivation.** ubuntu per-shard floor ≈ 39 s + 49 s install + numba warmup
  ≈ 2–3 min ≪ the N=3 ubuntu shard budget (~6.8 min) → **N=3 not floor-limited on ubuntu; Lever B not
  required.** Windows floor ≈ 17:48/3 + 1:49 install + numba ≈ **8:15**, margin ~1:45 → **binding leg;
  caching (Lever C + pip) prioritized there.** This is the record the spec's §8 criterion requires.

---

## Task 1: De-risk shardability BEFORE touching `ci.yml`

**Files:**
- Temporary/local only (a scratch venv with `pytest-split`); no committed change in this task.

**Interfaces:**
- Produces: a go/no-go on sharding — proof that the 3 shards partition the suite AND run green
  independently, and that `--co` reflects the per-group node set.

- [ ] **Step 1: Install `pytest-split` locally.** `pip install pytest-split`.
- [ ] **Step 2: Grep for the write-then-read cross-test hazard (spec §7, C7).** A test that writes a
  file another test reads breaks under cross-machine sharding. Run:
  ```bash
  grep -rnE "\.write_text|\.to_parquet|to_hdf|open\([^)]*['\"]w|SHA256SUMS|--store-durations|write_bytes" tests/ \
    | grep -vE "tmp_path|tmpdir|tmp_factory|scratch|/tmp|monkeypatch"
  ```
  For every hit that writes into `silly_kicks/**` or `tests/datasets/**` (not a `tmp_path`), confirm the
  reader is in the SAME test (or a fixture in the same module) — not a separate test that could land in a
  different shard. Record findings. A genuine cross-test file dependency is a bug to fix/isolate, not a
  reason to abandon sharding.
- [ ] **Step 3: Prove the partition + independence locally.** For each group run a FRESH process:
  ```bash
  for g in 1 2 3; do pytest tests/ -m "not e2e" --splits 3 --group $g -p no:randomly --benchmark-skip -q -p no:faulthandler; done
  ```
  Expected: all three green, and the three "collected N items" sum to the full `-m "not e2e"` collection.
- [ ] **Step 4: Verify `--co` reflects the per-group set** (the reconcile job depends on it):
  ```bash
  a=$(pytest tests/ -m "not e2e" --splits 3 --group 1 --co -q | grep -c "::")
  b=$(pytest tests/ -m "not e2e" --splits 3 --group 2 --co -q | grep -c "::")
  c=$(pytest tests/ -m "not e2e" --splits 3 --group 3 --co -q | grep -c "::")
  full=$(pytest tests/ -m "not e2e" --co -q | grep -c "::")
  echo "shards=$((a+b+c)) full=$full"   # MUST be equal
  ```
  If `--co` does not respect `--splits` (sum ≠ full), the reconcile job (Task 5) switches to capturing
  node IDs from a `-v`/`--report-log` run instead — decide here.
- [ ] **Step 5: Verify the numba on-disk cache POPULATES *and* RESTORES (P5).** Read
  `silly_kicks/tracking/pitch_control/_numba_kernels.py:20-45` — `_NUMBA_CACHE` is True iff a writable
  locator exists. (a) Confirm `NUMBA_CACHE_DIR=/tmp/nb pytest tests/tracking/test_numba_parity.py -q`
  populates `/tmp/nb` with `.nbi`/`.nbc` files. (b) **Confirm it RESTORES** — run a heavy-compile test
  (`tests/tracking/test_ghost_gk_integration.py` or a cover-shadow test) twice against the same
  `NUMBA_CACHE_DIR` and confirm the 2nd run does NOT recompile (numba logs, or wall-time drop). That
  restore is the property Lever C actually needs; populate-without-restore would be a silent no-op. (No
  source change — the kernels already declare `@njit(cache=_NUMBA_CACHE)`.) **P9 (low):**
  `NUMBA_CACHE_DIR: ${{ github.workspace }}/.numba_cache` yields a mixed-separator path on Windows
  (`D:\a\...\repo/.numba_cache`); numba tolerates it, but confirm the Windows restore in (b) rather than
  assuming — normalize only if it misbehaves.

---

## Task 2: Add `pytest-split` and generate `.test_durations`

**Files:**
- Modify: `pyproject.toml` (the `[test]` extra, lines 89–122)
- Create: `.test_durations` (repo root)
- Modify: `CLAUDE.md` (Testing section — the regen note)

**Interfaces:**
- Produces: `pytest-split` available in CI; a committed duration map for balanced splitting.

- [ ] **Step 1: Add the dep.** In `pyproject.toml` `[test]`, after `"pytest-benchmark>=4.0.0",`:
  ```toml
      # pytest-split shards the suite into duration-balanced groups across parallel CI jobs
      # (--splits/--group). Dev/test-only; NOT a runtime dependency. See ADR (CI parallelization)
      # + tests/test_ci_shard_wiring.py.
      "pytest-split>=0.9.0",
  ```
- [ ] **Step 2: Generate the durations file** from a full primary-selection run (ideally captured from
  a CI primary runner; local py3.12 acceptable for v1 — balance-only, never coverage):
  ```bash
  pytest tests/ -m "not e2e" --store-durations -p no:faulthandler --benchmark-skip -q
  ```
  This writes `.test_durations` (JSON, `pytest-split`'s default path) at repo root. Commit it in the
  SAME commit as `ci.yml` (spec §9 / nit 3 — else the debut run silently count-balances).
- [ ] **Step 3: Document the regen trigger** in `CLAUDE.md` Testing section (one line): regenerate
  `.test_durations` via the Step-2 command when the suite shifts materially or a shard drifts toward
  budget; balance is tuned for the ubuntu primary leg (windows/other-py may run hotter — acceptable, the
  reconcile guard still proves completeness).
- [ ] **Step 4: Confirm `.test_durations` is outside the lint/format scope** (repo root, not under
  `silly_kicks/`/`tests/`/`scripts/`) — `python -m ruff check silly_kicks/ tests/ scripts/` unaffected.

---

## Task 3: Rewrite the `ci.yml` `test` job into a sharded matrix

> **PREREQUISITE (P4, red-first):** do **Task 6 Steps 1–2 first** — author `test_ci_shard_wiring.py`
> and observe it FAIL against this (still un-sharded) `ci.yml`. Once this task rewrites `ci.yml`, the
> red observation is impossible. Same for the doctest/pandas-span tightening (write the assertion,
> watch it fail on the un-gated step, then gate).

**Files:**
- Modify: `.github/workflows/ci.yml` (the `test` job, lines 42–122)

**Interfaces:**
- Consumes: `pytest-split`, `.test_durations`.
- Produces: `N·4` sharded test jobs; per-leg shard-1 doctest + pandas-major; numba + pip caching.

- [ ] **Step 1: Add the `shard` axis + caching to the `test` job.** Replace the matrix + steps. Key
  changes (full YAML):
  ```yaml
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]
        shard: [1, 2, 3]          # N=3; kept consistent with --splits 3 by test_ci_shard_wiring.py
        exclude:
          - {os: windows-latest, python-version: "3.10"}
          - {os: windows-latest, python-version: "3.11"}
        include:
          - {os: ubuntu-latest, python-version: "3.12", primary: true}
    env:
      SILLY_KICKS_ASSERT_INVARIANTS: "1"
      NUMBA_CACHE_DIR: ${{ github.workspace }}/.numba_cache   # Lever C: persist compiled @njit kernels
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip                                          # wheel-download cache (subsequent-commit win)
      # Lever C — restore/persist numba's on-disk compiled kernels. Keyed on os+py+the @njit source
      # files so a kernel edit invalidates it. MOST impactful on the install/numba-bound windows leg.
      # The two hashFiles patterns cover ALL 4 current numba files -- 3 @njit-DECORATED (_numba_kernels,
      # _ghost_gk_numba, _ball_carrier_numba, via the glob) + 1 CALL-form njit (_turnover.py:96
      # `_njit(cache=True)(...)`, listed explicitly). This coverage is NOT left to the naming convention
      # -- `_turnover.py` already breaks it, and it is the call form the AST detector exists to catch --
      # it is PINNED by
      # test_ci_shard_wiring.py::test_numba_cache_key_covers_all_njit_files (P5), which fails CI if a new
      # @njit file is added without extending this key. A NARROW key (not silly_kicks/**/*.py) is
      # deliberate: it survives across PRs that touch non-numba code, which is what actually helps the
      # windows floor -- an over-broad key would cold-recompile on every source edit.
      - uses: actions/cache@<pinned-sha> # v4  (pin at implementation time, from the repo's SHA-pin convention)
        with:
          path: ${{ github.workspace }}/.numba_cache
          key: numba-${{ matrix.os }}-${{ matrix.python-version }}-${{ hashFiles('silly_kicks/tracking/**/*_numba*.py', 'silly_kicks/xtgk/_turnover.py') }}
      - run: pip install -e ".[kloppy,xgboost,das,test]"
      # Bulk suite, SHARDED. `--splits 3 --group <shard>` partitions the (marker-selected) suite into
      # duration-balanced groups from .test_durations; `-p no:randomly` pins collection order (the
      # partition is only valid if every shard collects identically). ADR-023 slow-gating preserved:
      # non-primary excludes @slow; primary runs everything.
      - if: ${{ !matrix.primary }}
        run: pytest tests/ -m "not e2e and not slow" --splits 3 --group ${{ matrix.shard }} -p no:randomly --benchmark-skip --tb=short --durations=25
      - if: ${{ matrix.primary }}
        run: pytest tests/ -m "not e2e" --splits 3 --group ${{ matrix.shard }} -p no:randomly --benchmark-skip --tb=short --durations=25
      # Per-leg-once steps: doctest (version-sensitive) + pandas-major, on shard 1 only.
      - if: ${{ matrix.shard == 1 }}
        run: pytest --doctest-modules silly_kicks/ --ignore-glob="*/_[!_]*.py" --tb=short
      - if: ${{ matrix.shard == 1 }}
        name: Record resolved pandas major
        run: python -c "import pandas, pathlib; pathlib.Path('pandas-major.txt').write_text(pandas.__version__.split('.')[0])"
      - if: ${{ matrix.shard == 1 }}
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: pandas-major-${{ matrix.os }}-${{ matrix.python-version }}
          path: pandas-major.txt
      # Node-ID lists for the reconcile job (Task 5). Each shard uploads its own; shard 1 also the full.
      # `shell: bash` is REQUIRED (P1): windows-latest defaults to pwsh, where `grep` is absent and
      # `sort` is Sort-Object -- the pipe would error and redden every Windows shard. Git Bash ships on
      # the Windows runners, so `shell: bash` makes the same pipe work on all legs (mirrors why the
      # pandas-major step uses python -c rather than a pipe).
      - shell: bash
        run: pytest tests/ -m "${{ matrix.primary && 'not e2e' || 'not e2e and not slow' }}" --splits 3 --group ${{ matrix.shard }} -p no:randomly --co -q | grep "::" | sort > shard-nodeids.txt
      - uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: shard-nodeids-${{ matrix.os }}-${{ matrix.python-version }}-${{ matrix.shard }}
          path: shard-nodeids.txt
      - if: ${{ matrix.shard == 1 }}
        shell: bash
        run: pytest tests/ -m "${{ matrix.primary && 'not e2e' || 'not e2e and not slow' }}" -p no:randomly --co -q | grep "::" | sort > full-nodeids.txt
      - if: ${{ matrix.shard == 1 }}
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: full-nodeids-${{ matrix.os }}-${{ matrix.python-version }}
          path: full-nodeids.txt
  ```
  (`fail-fast: false` is added so one shard failing does not cancel the others — you want the full picture.)
- [ ] **Step 1b (P6): Retain the four load-bearing comment blocks VERBATIM** from the current `test`
  job — they encode institutional knowledge this repo treats as first-class: the **DAS-on-every-leg**
  rationale (current lines 74–79), the **`xdist`-revert / memory-kill** rationale (82–86), the
  **ADR-023 slow-gating** block (88–94), and the **ADR-057 pandas-major** block (112–115). The
  abbreviated YAML above is illustrative; do NOT let a copy-paste delete these. Diff the old vs new
  `test` job and confirm each block survives (relocated as needed, not dropped).
- [ ] **Step 2: Move the benchmark step OUT** of the `test` job (it becomes Task 4's standalone job).
- [ ] **Step 3: Confirm the `--splits`/`--group`/`-m` ternary renders.** GitHub `${{ a && b || c }}`
  is the supported conditional-string idiom; verify the node-ID step's marker expression is quoted so
  YAML does not eat the `||`.

---

## Task 4: Standalone benchmark job

**Files:**
- Modify: `.github/workflows/ci.yml` (add a `benchmark` job)

- [ ] **Step 1: Add the job** (primary environment, parallel, off the critical path — spec N1):
  ```yaml
  benchmark:
    runs-on: ubuntu-latest
    env:
      SILLY_KICKS_ASSERT_INVARIANTS: "1"
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[kloppy,xgboost,das,test]"
      # Benchmark MEASUREMENTS (no hard timing asserts after the structural-guard conversion), single-
      # threaded for clean trend data. Standalone so it never sits on a shard's critical path.
      - run: pytest tests/ -m "not e2e" --benchmark-only --tb=short
  ```

---

## Task 5: `shard-reconcile` job — node-ID-set partition proof

**Files:**
- Modify: `.github/workflows/ci.yml` (add a `shard-reconcile` job)

**Interfaces:**
- Consumes: the `shard-nodeids-*` + `full-nodeids-*` artifacts from Task 3.
- Produces: a hard per-leg assertion `union(shards) == full ∧ pairwise-disjoint`.

- [ ] **Step 1: Add the job.**
  ```yaml
  shard-reconcile:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1
        with: {pattern: "shard-nodeids-*", path: shards/}
      - uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1
        with: {pattern: "full-nodeids-*", path: fulls/}
      - name: Shards must PARTITION each leg's suite (union == full, pairwise-disjoint)
        env:
          EXPECTED_N: "3"   # keep consistent with the shard axis; the wiring guard pins the axis to 1..N
        run: |
          python - <<'PY'
          import os, pathlib, sys
          N = int(os.environ["EXPECTED_N"])
          # artifact dir names: shard-nodeids-<os>-<py>-<shard> and full-nodeids-<os>-<py>.
          # leg = "<os>-<py>" (e.g. ubuntu-latest-3.12). <os> and <py> contain '-'/'.', so parse by
          # stripping the known prefix and (for shards) the trailing -<shard>, never by a fragile regex.
          def leg_shard(name): return name[len("shard-nodeids-"):].rsplit("-", 1)[0]
          def leg_full(name):  return name[len("full-nodeids-"):]
          # P2: node IDs can contain spaces (parametrized: `test[a b]`), so read one-per-LINE, never
          # .split() on whitespace. The files are grep "::"-filtered already; keep only `::` lines.
          def load(p): return {ln for ln in p.read_text().splitlines() if "::" in ln}
          shards, fulls = {}, {}
          for f in pathlib.Path("shards").rglob("shard-nodeids.txt"):
              shards.setdefault(leg_shard(f.parent.name), []).append(load(f))
          for f in pathlib.Path("fulls").rglob("full-nodeids.txt"):
              fulls[leg_full(f.parent.name)] = load(f)
          # P3: non-vacuity on BOTH sides + leg-set symmetry -- an empty download must FAIL, not print
          # "OK: 0 legs" (pandas-span's "pass having observed NOTHING" is the bar this must clear).
          assert fulls,  "::error::no full-nodeids artifacts -- reconcile would pass vacuously"
          assert shards, "::error::no shard-nodeids artifacts -- reconcile would pass vacuously"
          errs = []
          if set(shards) != set(fulls):
              errs.append(f"leg mismatch: shards={sorted(shards)} fulls={sorted(fulls)}")
          for lg in sorted(set(shards) | set(fulls)):          # iterate the UNION so a one-sided leg is caught
              groups, full = shards.get(lg), fulls.get(lg)
              if groups is None: errs.append(f"{lg}: full present but NO shard artifacts"); continue
              if full is None:   errs.append(f"{lg}: shard artifacts present but NO full to compare"); continue
              if len(groups) != N: errs.append(f"{lg}: {len(groups)} shard artifacts, expected {N} (a shard is missing)")
              seen = set()
              for g in groups:                                  # pairwise-disjoint
                  dup = seen & g
                  if dup: errs.append(f"{lg}: {len(dup)} node(s) in >1 shard e.g. {sorted(dup)[:3]}")
                  seen |= g
              missing, extra = full - seen, seen - full
              if missing: errs.append(f"{lg}: {len(missing)} test(s) in NO shard e.g. {sorted(missing)[:3]}")
              if extra:   errs.append(f"{lg}: {len(extra)} test(s) not in full set e.g. {sorted(extra)[:3]}")
          if errs:
              # P8: a red here can also mean cross-runner COLLECTION DIVERGENCE (a module-top importorskip
              # resolving differently on one machine after an install flake) -- that is the INTENDED catch
              # (spec §6b). Diagnose the diverging node IDs; never "fix" it by weakening this guard.
              sys.exit("::error::shard partition broken (or cross-runner collection divergence):\n" + "\n".join(errs))
          print(f"OK: {len(fulls)} legs, {N} shards each partition the full suite exactly")
          PY
  ```
- [ ] **Step 2: Keep `pandas-span` unchanged** (`needs: test` still waits for all shards; artifacts come
  from each leg's shard 1).

---

## Task 6: Wiring guards — new + tightened

**Files:**
- Create: `tests/test_ci_shard_wiring.py`
- Modify: `tests/test_ci_doctest_wired.py` (assert shard-1 gating)
- Modify: `tests/test_ci_pandas_span_wired.py` (assert record/upload is shard-1-gated)

**Interfaces:**
- Consumes: `ci.yml` (parsed), `pyproject.toml`.
- Produces: static pins on the shard config, determinism, and shard-1 gating.

- [ ] **Step 1: Write `tests/test_ci_shard_wiring.py` (failing first).** Assertions:
  ```python
  """Structural guard: the sharded CI matrix is internally consistent and deterministic.

  Sharding is only a valid PARTITION if (a) the shard axis is contiguous 1..N, (b) --splits N matches
  N, (c) collection order is pinned (no shuffle plugin), so no test silently runs in zero shards. This
  is the pre-flight complement to the runtime shard-reconcile job. See the CI-parallelization ADR.
  """
  import ast, pathlib, re, yaml, tomllib
  _REPO = pathlib.Path(__file__).resolve().parent.parent
  _CI = yaml.safe_load((_REPO / ".github/workflows/ci.yml").read_text())

  def _defines_njit(src: str) -> bool:
      # P10: detect BOTH @njit(...) / @numba.njit(...) decorators AND the njit(...)(fn) CALL form
      # (_turnover.py). AST, not a regex, so a docstring mention of "@njit" is never a false positive
      # (ADR-056: read code, not prose).
      try: tree = ast.parse(src)
      except SyntaxError: return False
      NJIT = {"njit", "_njit"}
      for n in ast.walk(tree):
          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
              for d in n.decorator_list:
                  f = d.func if isinstance(d, ast.Call) else d
                  nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                  if nm in NJIT: return True
          if isinstance(n, ast.Call) and isinstance(n.func, ast.Call):      # njit(...)(fn)
              inner = n.func.func
              nm = inner.attr if isinstance(inner, ast.Attribute) else getattr(inner, "id", None)
              if nm in NJIT: return True
      return False

  def _sharded_cmds():
      return [s["run"] for s in _CI["jobs"]["test"]["steps"]
              if "run" in s and "--splits" in s["run"] and "pytest tests/" in s["run"]]

  def test_shard_axis_is_contiguous_1_to_N():
      shards = _CI["jobs"]["test"]["strategy"]["matrix"]["shard"]
      assert shards == list(range(1, len(shards) + 1)), f"shard axis must be 1..N contiguous, got {shards}"

  def test_splits_value_matches_shard_count():
      n = len(_CI["jobs"]["test"]["strategy"]["matrix"]["shard"])
      for cmd in _sharded_cmds():
          m = re.search(r"--splits\s+(\d+)", cmd)
          assert m and int(m.group(1)) == n, f"--splits must equal shard count {n}: {cmd}"
          assert "--group ${{ matrix.shard }}" in cmd, f"missing per-shard --group: {cmd}"

  def test_every_sharded_command_pins_collection_order():
      for cmd in _sharded_cmds():
          assert "-p no:randomly" in cmd, f"sharded command must pin collection order: {cmd}"

  def test_no_collection_shuffling_plugin_in_test_extra():
      pyproj = tomllib.loads((_REPO / "pyproject.toml").read_text())
      test_deps = pyproj["project"]["optional-dependencies"]["test"]
      assert not any("pytest-randomly" in d for d in test_deps), (
          "pytest-randomly auto-activates and would break the shard partition; keep it out of [test]"
      )

  def test_benchmark_is_a_standalone_job_not_on_a_shard():
      assert "benchmark" in _CI["jobs"], "benchmark must be its own parallel job (spec N1)"
      test_runs = " ".join(s.get("run", "") for s in _CI["jobs"]["test"]["steps"])
      assert "--benchmark-only" not in test_runs, "benchmark-only must NOT sit on a sharded test step"

  def test_shard_reconcile_job_exists_and_needs_test():
      job = _CI["jobs"].get("shard-reconcile")
      assert job is not None and job["needs"] == "test"

  def test_numba_cache_key_covers_all_njit_files():
      # P5/P10: every @njit file (decorator OR call form) must be in the numba actions/cache hashFiles(),
      # or its recompiled blob is never saved (cache HIT => no upload) and Lever C silently no-ops for it
      # -- worst on the binding windows leg. The `*_numba*` naming is NOT relied on (`_turnover.py` breaks
      # it, via the call form); this pins coverage so a NEW @njit file fails CI until the key is extended.
      njit_files = sorted(
          str(p.relative_to(_REPO)).replace("\\", "/")
          for p in (_REPO / "silly_kicks").rglob("*.py")
          if _defines_njit(p.read_text(encoding="utf-8"))
      )
      # non-vacuity (ADR-056): the detector must still find the awkward call-form file it exists for.
      assert "silly_kicks/xtgk/_turnover.py" in njit_files, (
          "detector no longer finds the call-form njit file (_turnover.py) -- it has drifted"
      )
      cache = [s for s in _CI["jobs"]["test"]["steps"] if "actions/cache" in str(s.get("uses", ""))]
      assert cache, "no numba actions/cache step in the test job"
      patterns = re.findall(r"'([^']+)'", str(cache[0]["with"]["key"]))
      # Path.glob DOES treat ** as zero-or-more dirs, matching GitHub hashFiles -- fnmatch does NOT
      # (it would false-fail files sitting directly under tracking/). P10.
      covered = set()
      for pat in patterns:
          covered |= {str(p.relative_to(_REPO)).replace("\\", "/") for p in _REPO.glob(pat)}
      missing = set(njit_files) - covered
      assert not missing, f"@njit files not covered by the numba cache key: {sorted(missing)}"
  ```
- [ ] **Step 2 (RED-FIRST — do BEFORE Task 3, P4): author this guard and observe it FAIL against the
  CURRENT (un-sharded) `ci.yml`.** `test_shard_axis_is_contiguous_1_to_N` (no `shard` key),
  `test_splits_value_matches_shard_count` (no `--splits` cmd), `test_shard_reconcile_job_exists...`
  (no such job) all go red. This is the "detection lands before the fix" gate — it is impossible to
  observe once Task 3 has rewritten `ci.yml`, so **Task 6 Steps 1–2 execute before Task 3, and Steps
  3–4 (tightening) likewise write-the-assertion-then-watch-it-fail on the still-un-gated step before
  adding the `shard == 1` gate.** After Task 3+4+5 land, re-run the whole guard suite → green.
- [ ] **Step 3: Tighten `test_ci_doctest_wired.py`.** Add: the single doctest step is gated on
  `matrix.shard == 1` (runs once per leg, not N×) — a new assertion alongside the existing
  `matrix.primary not in guard`:
  ```python
  def test_doctest_runs_once_per_leg_on_shard_1():
      guard = _guard(_doctest_steps()[0].get("if", ""))
      assert "matrix.shard==1" in guard.replace("'", "").replace('"', ""), (
          f"doctest must be gated on shard 1 (once per leg), got {guard!r}"
      )
  ```
- [ ] **Step 4: Tighten `test_ci_pandas_span_wired.py`.** Extend
  `test_every_test_leg_records_its_pandas_major` to require the record + upload steps are gated on
  `matrix.shard == 1` — else N shards per leg upload the same artifact name and `upload-artifact@v4`
  fails on the duplicate. Assert the `if:` on the record and upload steps contains `matrix.shard==1`.

---

## Task 7: End-to-end validation on real CI (acceptance gate)

**Files:** none (measurement).

**Interfaces:**
- Produces: the measured proof of the SLA — the only place `< 10 min` is real.

- [ ] **Step 1: Local full-guard pass** — `pytest tests/test_ci_*_wired.py tests/test_ci_shard_wiring.py -q` green; `ruff` + `pyright` clean.
- [ ] **Step 2: After the owner approves + pushes** (see rollout), read the run:
  ```bash
  gh run list --workflow=ci.yml --limit 1 --json databaseId -q '.[0].databaseId'
  gh run view <id> --json jobs -q '.jobs[] | "\(.name) \(.startedAt)->\(.completedAt)"'
  ```
  Assert: the SLOWEST job (expected windows shard) completes **< 10 min** from run start; `shard-reconcile`
  and `pandas-span` green; no job queued for lack of concurrency.
- [ ] **Step 3: If a leg exceeds 10 min, the tuning lever depends on WHICH leg (P7 — they bind for
  different reasons):**
  - **Windows over** (the likely case; install/numba-bound, 1:49 install is undivided): (a) confirm the
    **numba cache actually RESTORES** on the 2nd/warm run (not just populates) for a ghost-GK/cover-shadow
    test; (b) confirm **pip cache** restores; (c) escalate lever **(B-Windows)** to the owner. **Do NOT
    bump N for a Windows miss** — `17:48/3 → /4` shaves only the *test* term while the undivided install +
    numba dominate, so an N-bump barely moves it.
  - **An ubuntu leg over** (test-bound): bump `N` to 4 (edit BOTH literals — the guard enforces
    consistency — and re-check ~14→~18 peak-concurrent against the 20 cap), or apply a Lever-B
    coverage-preserving fix to the tail.
  - **Never silently drop coverage to hit the number** (spec §2 N3 / Task 7 guardrail).

---

## Task 8: TODO note + rollout wiring

**Files:**
- Modify: `TODO.md` (record the deferred lever)
- Reference: the ADR (drafted at `/final-review` time per the repo convention) + version bumps at commit-prep.

- [ ] **Step 1: Add a `TODO.md` On-Deck/Tech-Debt line** for the deferred **lever (B-Windows)** — trim
  Windows to the platform-/version-sensitive subset as the next latency/cost lever — and note the
  `.test_durations` manual-regen trigger. (Delete-don't-annotate grooming; no release history in On-Deck.)
- [ ] **Step 2: At `/final-review`**, draft the CI-parallelization ADR (the CI structure is a
  convention with downstream effect — sharding, `.test_durations`, the reconcile contract) and confirm
  the `test_ci_*_wired` + shard-wiring guards are green.
- [ ] **Step 3: Commit-prep (owner-approved only):** commit 2 = all of the above **with
  `.test_durations`**; version bump per convention; numbers at commit-prep from merged `origin/main`.
  A 3rd commit only if Lever B produced test edits.

---

## Self-review notes (author)

- **Spec coverage:** Task 0 = §2 measurement; Task 1 = §7 shardability probe; Task 2 = §4 durations +
  `pytest-split`; Task 3 = §3/§4/§5 sharded matrix + caching; Task 4 = §3 N1 standalone benchmark;
  Task 5 = §6b node-ID reconcile; Task 6 = §6a + doctest/pandas guard tightening + B3 determinism;
  Task 7 = §8 SLA acceptance; Task 8 = §9 rollout + deferred B-Windows. All spec sections mapped.
- **The three `--co` node-ID steps in Task 3 depend on Task 1 Step 4 confirming `--co` respects
  `--splits`.** If it does not, Task 5 falls back to `--report-log` capture — flagged in Task 1.
- **`actions/cache` SHA is left as `<pinned-sha>`** deliberately — pin it at implementation time from the
  repo's existing pinned-action convention (every Action in the repo is SHA-pinned); do not use a
  floating `@v4`.
