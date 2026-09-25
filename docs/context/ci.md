# CI, testing internals & gate discipline

> On-demand class-2 context (ADR-102-style store). The current terse RULES live in `AGENTS.md`; this file holds the WHY / history / measurement. See the spec `docs/superpowers/specs/2026-09-24-agents-md-restructure-design.md`.

- **Every `add_*` carries an SB360 freeze-frame verdict, and the OBSERVATION is locked while the ADJUDICATION is not (ADR-053).** `tests/sb360/` runs each aggregator on a paired fixture -- Leg A built by the real `snapshot_to_tracking_frames`, Leg B velocity-bearing with identical positions at the linked frame -- and records a per-COLUMN, per-AXIS verdict. CI re-derives the machine observation (`identical`/`differs`/`all_nan`/`partial_nan`/`no_signal`/`raises_a`) and asserts it; the human adjudication (`works`/`silent_degrade`/`differs_by_design`/`honest_nan`/`not_exercised`/`raises`) carries a mandatory rationale and is NOT locked, because a machine cannot tell *fabricated* from *legitimately different* -- pitch control at zero velocity is a valid positional model, a fitted model silently imputing the features it was trained on is not (`add_ghost_gk` WAS that case: the extractor yields NaN and the HGBR routes it down a LEARNED missing-value branch, which is not a zero-fill -- the verdict was right while the first-recorded mechanism was wrong, which is exactly why the adjudication stays reviewable). **REPAIRED: the ghost path now REFUSES on the velocity marker at the shared serving seam `_serve_positions_core`, so all three public entry points inherit it, and the audit re-derives to ZERO `silent_degrade` -- the positions moved to `honest_nan` by rule, not by hand-edit.** The example survives as the case that motivated the split, not as a live defect. Locking the verdict instead was tried and rejected: it pins the key set while the content rots. A new `add_*` must register or CI fails. Registry regenerable via `tests/sb360/_regenerate.py` + `_adjudicate.py` (round-trip verified byte-identical). **Boundary entry points outside `tracking.__all__` (xtgk v2 `compute_xt_gk_v2` / the three `gkdv` arms / `spadl.add_restart_coordinates`) ALSO carry SB360 verdicts + a per-entry `verdict_provenance` (`substantive`/`structural`); `UNAUDITABLE_BOUNDARY` is EMPTY (ADR-053 amendment, 4.88.0).** The `test_boundary_entries_declare_admissible_provenance` gate LOCKS the frame-blind half (`works` -> `structural`, e.g. `compute_xt_gk_v2` -- a value that cannot move across the velocity legs was not substantively handled) and AUTHOR-ASSERTS the observationally-ambiguous `honest_nan` half (self-refusal vs inherited-refusal both produce `all_nan`, so gkdv's `structural` is forced only to carry a rationale, not a token). A new boundary entry must register + declare provenance or CI fails; `honest_nan`/inherited entries (gkdv) inherit the ADR-054 ghost-serving refusal and the arms' intrinsic zero-velocity behaviour stays out of scope.

- **Detection lands BEFORE the fix, and every registry gate carries an anti-rot meta-assertion (ADR-051).** A gate written after its own repair arrives green and is never observed failing, so the mirror gates above were landed RED -- enumerating all 33 aggregators before a single fix existed. Both pin their registry to `tracking.__all__` in BOTH directions, so a new `add_*` must register or CI fails. Corollary to Gate A's blindness (see the ADR-028 bullet): a clean Gate A reading is NOT evidence about identity-keyed direction -- one such reading was withdrawn on exactly that ground.


```bash
python -m pytest tests/ -m "not e2e" -v --tb=short
```

e2e tests require dataset fixtures not committed to the repo. Tests with
fixtures committed to the repo should not be marked e2e — they run in
the regular suite.

**Lint at the CI scope, never `.`** — `python -m ruff check silly_kicks/ tests/ scripts/` and
`python -m ruff format --check silly_kicks/ tests/ scripts/`, matching `ci.yml`. `ruff check .`
walks `.venv/` and `calibration_runs/` and reports ~234 vendored errors that are not the repo's,
which is enough noise to hide the real ones. `pyright` runs **bare** (config-driven include), and
neither tool is on PATH — use `python -m`.

CI **duration-shards** the bulk suite across parallel jobs via `pytest-split` (ADR-074)
(`--splits 3 --group ${{ matrix.shard }}`, matrix `os × python × shard[1..3]`), each shard on its
own runner (the `xdist -n auto` memory-kill on the 4-core/7GB runners is why intra-job parallelism
was reverted — sharding gives the parallelism without the shared-memory contention). The split is
balanced by the committed **`.test_durations`**; **without it pytest-split's count-mode split is
NON-DETERMINISTIC (measured: shard sizes drift run-to-run and can under-cover), so `.test_durations`
is committed.** The committed file is **CI-MEASURED** (a local `--store-durations` mis-balances — CI
is ~2× local and per-py-version relative timings differ; measured: a local-durations shard ran 11:26
vs 5:50 on CI, whereas the CI-measured file balances all three primary shards to ~7.3 min). **To
regenerate:** temporarily re-add a `durations-capture` job (full `pytest -m "not e2e"
--store-durations` under the warm numba cache, `include-hidden-files: true` on the upload since
`.test_durations` is a dotfile), download its `test-durations-ci` artifact, commit it, and remove the
job (a permanent ~20-min serial capture would become the wall-clock bottleneck). Regenerate when the
suite shifts materially or a shard drifts toward budget; balance is tuned for the ubuntu primary leg
(others may run hotter — acceptable, the runtime `shard-reconcile` job still proves completeness). `-p no:randomly` pins collection order (a shuffle plugin would break the partition;
`tests/test_ci_shard_wiring.py` bans it). Coverage is proved two ways: the static
`tests/test_ci_shard_wiring.py` (contiguous `1..N`, `--splits == N`, `-p no:randomly`, numba-cache
key covers all `@njit` files) and the runtime `shard-reconcile` job (node-ID `union == full ∧
pairwise-disjoint`, per leg). Benchmark *measurements* run single-threaded in a **standalone**
`benchmark` job (off the shard critical path). Windows is the binding leg (undivided ~1:49 install);
numba (`NUMBA_CACHE_DIR` + `actions/cache`) + pip caching are prioritized there. There are no
wall-clock `assert ms < budget` perf tests — performance regressions are guarded by **deterministic
structural guards** (call-count spies; `tests/_perf_structural.py` + the `*_perf_budget.py` files).

**CI's pandas-major span is DECLARED, not inherited (ADR-057).** The matrix is OS × Python with no
pandas axis, yet it spans both majors: `pyproject.toml` pins `pandas>=2.1.1,!=3.0.4` with **no upper
bound** and pandas 3 requires Python ≥3.11, so pip resolves the newest compatible pandas per
interpreter — measured, `ubuntu-3.10` → **2.3.3** and the other three legs → **3.0.5**. That coverage
was real but ACCIDENTAL, and could have vanished with no diff and no signal (this repo has one
measured instance of that class: DAS going silently all-NaN on pandas 3). It is now asserted by a
PAIR, because a test running in one leg cannot observe another: `tests/test_ci_pandas_span_wired.py`
pins that the **resolved leg set** still straddles the 3.11 boundary — parsing os × python-version
**minus `exclude` plus `include`**, never the axis, since `exclude` is the pruning mechanism already
in use and excluding a leg collapses the span while leaving its version in the axis — and a
`pandas-span` job (`needs: test`) asserts the union of each leg's recorded major covers both. Assert
the SPAN, never specific versions: pinning "3.10 → 2.3.3" fails on every routine bump and trains a
reader to edit the expectation without thinking. **A `.to_numpy()` result you intend to MUTATE needs
`copy=True`** — under pandas 3's copy-on-write it is a READ-ONLY view when no dtype conversion is
required, so `arr[mask] = x` raises `ValueError: assignment destination is read-only`; pandas 2
returned a writable array, which means a full local suite on the 3.10 leg passes and sees nothing
(measured: 48 failures on 3.11, 0 locally, 4.80.0). Sweep this by PREDICATE, not by patching the
line that failed — an AST search for "a `to_numpy()` result later subscript-assigned" finds every
site repo-wide, and there are four. **Never assert a dtype literal across pandas
majors** — assert the behaviour (that `id_compat` comparisons still match), which is what made the
two-cycle-old `snapshot_to_tracking_frames` question finally answerable.

**A TYPING input is pinned; a BEHAVIOURAL input is not (4.80.0).** The lint job exact-pins
`ruff`, `pyright`, `pandas-stubs` **and `numpy`** -- numpy ships inline types, so it decides what
pyright REPORTS exactly as pandas-stubs does, and leaving it floating made the gate's verdict a
function of resolution luck (measured: main resolved 2.5.2 then DOWNGRADED to 2.4.6, so a PR whose
diff could not reach the three offending files went red while main stayed green). **Pin it on the
LAST install that can move it** -- the tools line runs first, but `pip install -e ".[test]"`
re-resolves, so a pin above it is decorative. The TEST matrix stays UNPINNED on purpose: there,
floating numpy/pandas is ADR-057's span and is what catches real breakage. Both halves are pinned
by `tests/test_ci_lint_pins_wired.py`, including the asymmetry, so "tidying" the jobs into
consistency fails CI rather than silently deleting the coverage.

**The BUILD BACKEND is bounded and the PUBLISHER's validator is pinned — the pair, or neither works
(4.79.0).** Same shape as the span above, and it bit within hours of that ADR landing: every Action
in `.github/workflows/` is SHA-pinned while `[build-system] requires` was `["hatchling"]` with **no
upper bound**, so the artifact's `Metadata-Version` was a function of WALL-CLOCK TIME. hatchling
1.32.0 (2026-08-11 05:03Z) moved 2.4 → 2.5; `v4.78.0` built at 01:39Z and published, `v4.79.0` built
at 12:56Z and was REFUSED by the publisher's pinned `packaging==25.0`, whose valid set stops at 2.4
— **green build, green CI, failed upload, no diff between the two runs.** Fix BOTH sides and bump
FORWARD (`gh-action-pypi-publish` v1.14.2 = `packaging==26.2` + `twine==7.0.0`): metadata 2.5 is
legitimate, and pinning the backend below it only defers the failure. Bounded (`>=1.27,<2`), never
pinned — the point is that a backend release lands in a DIFF instead of in a failed publish.
`tests/test_ci_publish_guard_wired.py` rejects any unbounded `[build-system]` requirement; because a
bound-checker passes vacuously if its predicate says yes to everything, a companion test pins that
the predicate rejects the exact `"hatchling"` string that broke this release. **The durable rule: a
pinned consumer and a floating producer are one dependency, and pinning only the half you control is
not a pin.** Corollary for diagnosis: I first argued v1.14.2 wouldn't help because its RELEASE DATE
predates hatchling 1.32.0 — the wrong lens entirely; what decides it is the validator version the
action pins, which is readable in seconds from `requirements/runtime.txt` at that tag.

**Doctests: CI executes `--doctest-modules` on the PUBLIC surface only** (`pytest --doctest-modules
silly_kicks/ --ignore-glob="*/_[!_]*.py"`, every leg; PR-S124, wiring guarded by
`tests/test_ci_doctest_wired.py`). The glob skips single-underscore
private modules while KEEPING dunder `__init__.py`; private-module examples are kept CORRECT (the
whole-package sweep is clean) but are NOT executed in CI, to bound wall-clock — the initial
enforcement scope. Most public examples are the canonical indented RST **literal block** (the honest
form for anything needing a real match's `actions`/`frames`/`xt` — no docstring can conjure them);
`tests/test_public_api_examples.py::_has_real_example` accepts a ≥4-space-indented non-`>>>` line in
the Examples section as a REAL example, so converting a failing `>>> f(actions, ...)` doctest to a
literal block satisfies BOTH the doctest run (nothing to execute) and the Examples gate. Genuinely
self-contained examples stay executable `>>>` doctests. Do NOT reintroduce `>>> f(x)  # doctest: +SKIP`
filler — the gate's `_demonstrates_something` rejects it (the 4.53.0 tightening).

**`@pytest.mark.slow` = expensive AND platform/interpreter-INVARIANT** (does-it-run train-script
smokes, same-run internal-consistency/parity, calibration cache-equivalence). These run **once on the
CI primary leg (`ubuntu-3.12`, matrix `primary: true`)**; every other leg runs `-m "not e2e and not
slow"`. Do **NOT** mark version-sensitive tests `slow` (golden-hash / snapshot / absolute-numeric — e.g.
`test_golden_*`) — they must stay on all legs (OS + interpreter axes), as must cheap behavioral-contract
guards (dup-`action_id`, id-dtype-invariance, orientation/roster). The matrix partition is guarded by
`tests/test_ci_slow_gating_wired.py`. Decision: ADR-023.

**Claims about a gate's behaviour must quote the assertion body, not its registration.**
A gate's registry (`ENTRIES`, `PURITY_ENTRIES`, `AGGREGATORS`, `_PUBLIC_MODULE_FILES`,
`_EXHAUSTIVE_EMITTED`) tells you *whether* a helper is exercised — never *what is asserted
about it*. Two PR-S119 review errors came from this, one on each side: a reviewer read the
liveness gate's registration and concluded a partially-`pd.NA` column would fail, when
`tests/tracking/test_aggregator_column_liveness.py:416` asserts only
`not out[c].notna().any()` (100%-null), and `:448`'s non-constant check is
float-dtype-gated so non-numeric columns are exempt entirely; and a plan asserted the
synthetic reachability grid was exactly y-symmetric and that symmetrizing it would cost
"≤ 1 ulp", when neither had been run (measured: 4.44e-16 across 3752/6400 cells). Same
failure mode both times — adjacent evidence substituted for the executed thing. **Rule: any
claim that a test passes or fails, or that a numeric property holds, carries either the
quoted assertion body or a pasted measurement.** Registration proves coverage; only the
assertion proves behaviour. Corollary for tolerances: measure at the scale you assert at —
grid-level float noise does not transfer to a derived area (a 3.3e-16 relative grid
asymmetry is not a 1e-12 absolute bound on an m² quantity).
