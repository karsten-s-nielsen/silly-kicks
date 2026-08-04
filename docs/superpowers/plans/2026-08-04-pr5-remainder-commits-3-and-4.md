# PR 5 remainder — commits 3 and 4 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Rev 5**, 2026-08-04. Six review rounds. The last one built rev 4's fixture, passed all four
gates, and returned **no blockers** — but established two things that made the plan SMALLER. First,
the forged-half hole is unclosable and harmless: the point reflection IS the goal-relative
transform, so a forged half holds the correct values and no assertion at any tolerance can find a
defect that is not there. Second, the fourth gate duplicated `test_pr5_chirality_gates.py`, which
already covers both extractors, both axes and a permanent plant, and extracts LIVE so it cannot be
forged at all. Rev 5 therefore DROPS the fourth gate and widens `pr5_scene()` instead — which also
retires the two-goal-end sweep, the pairing requirement and the rows-versus-distinct-vectors
reconciliation, since all three existed only to serve it.

**Goal.** Close PR 5 with every artifact it invalidated refreshed, every number backed by a
driver-produced file, and no known-stale artifact reaching `main`.

**Architecture.** Two commits with a hard boundary. Commit 3 ships **code and docs only** and runs
**no guarded driver at all**. Commit 4 runs every driver from the resulting clean tree, writes all
output **outside the repo**, copies it in once, and only then edits prose. The boundary is mechanical,
not stylistic: `scripts/_provenance.py:76` counts untracked files as dirty and `:92` raises
`SystemExit`, so **a driver cannot run in the same commit that introduces it** — writing the driver is
itself what makes the tree dirty.

**Tech stack.** pandas / numpy / scikit-learn, pytest, ruff, pyright, XGBoost (inference only), DGX
Spark (aarch64) for corpus runs, HuggingFace Hub for published variants.

**Source spec:** `docs/superpowers/specs/2026-08-03-adr051-closeout-and-artifact-validity-design.md`
rev 3 (+ the §8/§9 item-14 correction). Supersedes Tasks 14–15 of
`docs/superpowers/plans/2026-08-02-pr5-chiral-goal-relative-transform.md`.

## Global Constraints

- **Lint at the CI scope, never `.`**: `python -m ruff check silly_kicks/ tests/ scripts/`,
  `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright` (bare). Neither is
  on PATH — use `python -m`. `ruff check .` walks `.venv/` and reports ~234 vendored errors.
- **NEVER invoke a `scripts/*.py` that has no argument parser — not even with `--help`.** Without a
  parser `--help` is *ignored* and `main()` runs. Check `grep -q add_argument` first. **Task 5 adds a
  parser to the fixture generator rather than taking an exception**, because an exception in a plan is
  a rule someone else will break.
- **Every driver run writes OUTSIDE the repo** (`--out "$HOME/pr5_runs/<name>"`). This is not tidiness:
  an in-repo output dirties the tree and the *next* driver refuses. Rev 1 violated this in four places.
- **No guarded driver runs before commit 3 is committed and pushed.**
- **Never read a gate verdict through `| tail`** — redirect to a file and grep by name.
- **Merge with `--merge`, NEVER squash.** `6e3a132` and `08ce9a8` are cited as `run_commit` in shipped
  artifacts; a squash orphans every citation.
- **No version number until commit-prep**, after `git fetch origin && git merge origin/main`.
- **`run_commit` is never restamped without a re-run.**
- Constants: `FIELD_LENGTH = 105.0`, `GOAL_Y = 34.0`, `PITCH_WIDTH = GOAL_Y * 2.0`. There is **no**
  `FIELD_WIDTH`.

## Facts resolved before this plan — do not re-derive

1. **The xCross substitution probe needs NO re-run.** Its result is in
   `silly_kicks/tracking/_xcross_weights/default/metrics.json` with `run_commit: 6e3a132b0e75…`,
   `run_tree_dirty: false`, `gk_median_abs_delta 0.003582…`, `ratio 1.7013`, `tf19_ready false`.
   Item 17 is a **citation fix** (Task 12). **But the CHANGELOG's *before* leg (`1.41x`, `0.002417`)
   lives at `git show 6e3a132:<same path>`, which carries NO provenance** — Task 12 handles both legs.
2. **Commit 3 cannot stale the completed xS probe** — provided commit 3 excludes the version bump
   (Task 7). Task 11 verifies this over all of `silly_kicks/`, not three paths.
3. **The xS probe is COMPLETE**: 64/64 matches, 0 excluded, `v1=no_valid_placebo v2=pass
   regate_v2=joins_with_caveat lock=78ffc70` — identical to the pre-PR-5 result. Output is at
   `karsten@192.168.68.73:~/Development/pr5_runs/xs_probe/` (outside the repo). **The DGX tree is
   clean at `08ce9a8`** ✅. No waiting, no probe re-run.
4. **PR 5 ships NO taxonomy label.** Zero weight artifacts carry `artifact_label`/`all_public` ✅;
   `_corpus_taxonomy` exists only in `train_gk_completion.py`, which PR 5 never runs. Task 14 is a
   question, not a tag gate.
5. **`_geometry.py`'s commit-1 change is purely ADDITIVE** — new `to_goal_relative_y`/`_vy` plus a
   `GEOMETRY_VERSION` bump; no existing function changed behaviour. Task 3's baseline is therefore not
   contaminated *today*; the subprocess fix is adopted for **reuse**, and the reason is recorded so a
   later reader does not find a false justification and revert it.

## File structure

| File | Responsibility | Commit |
|---|---|---|
| `scripts/validate_xcross_causal.py` | provenance wiring (modify) | 3 |
| `scripts/measure_covariate_invariance.py` | **new** — two-axis covariate table (§4.5) | 3 |
| `scripts/measure_platform_probe.py` | **new** — one self-provenanced probe JSON per platform (§4.10) | 3 |
| `scripts/make_xcross_directional_fixture.py` | argparse + varied GK + in-domain + ≥40 rows (modify) | 3 |
| `tests/datasets/tracking/xcross_directional/frozen_rows.parquet` | regenerated fixture | 3 |
| `tests/tracking/test_xcross_attempt_integration.py` | retire 1 gate, add 3 (modify) | 3 |
| `tests/scripts/test_provenance_wiring.py` | enrol 3 drivers (modify) | 3 |
| `CLAUDE.md`, `docs/superpowers/adrs/ADR-051-*.md` | caller-sweep rule; close the deferral (modify) | 3 |
| `docs/research/*/` | all driver outputs | 4 |
| `CHANGELOG.md`, `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock` | version + narrative | 4 |

**The five version files are already modified in the working tree.** They belong to commit 4. Task 7
sets them aside; nothing before that may `git add -A`.

---

# COMMIT 3 — code and docs only. No driver runs.

### Task 1: Clean the tree, land the two docs the cycle owes

**Files:** delete `docs/research/pr5_platform_atol/`; modify `CLAUDE.md`,
`docs/superpowers/adrs/ADR-051-*.md`, `CHANGELOG.md`

- [ ] **Step 1: Move the hand-run measurement out of the repo**

```bash
SCRATCH="$HOME/pr5_scratch"; mkdir -p "$SCRATCH"
mv docs/research/pr5_platform_atol "$SCRATCH/pr5_platform_atol_handrun"
git status --porcelain docs/research/          # expect: empty
```
It is untracked (`git ls-files` returns 0), so this is a plain `mv`. Committing it and later adding a
`run_commit` would be the restamp §4.5 forbids.

- [ ] **Step 2: Add the caller-sweep rule to CLAUDE.md `## Key conventions`**

```markdown
- **A spec that changes a public seam enumerates every caller of every CHANGED FUNCTION and classifies
  each as affected or not WITH EVIDENCE ON BOTH SIDES -- and the sweep is the FLOOR, not the check.**
  Measured: the sweep that closed PR 5's blast radius took under a minute, returned five call sites and
  resolved two questions in opposite directions. It also MISSED four things, which is why the rule names
  them. A symbol sweep cannot see (a) a changed function you did not think to grep -- PR 5 changed FOUR
  and the sweep covered three, missing the `_dominant_region_area` grid re-anchor entirely; (b) wrapper
  callers (`derive_opengoal_range.py` imports `prepare_xshot_training_data`, which wraps the changed
  extractor); (c) committed data artifacts, which call nothing -- a frozen fixture went stale in
  `space_controlled` and no sweep could say so; (d) numbers recorded in prose rather than in a
  driver-produced file; (e) the second hop -- the sweep returns a module, never the research artifacts
  downstream of it. So ALSO enumerate: every function in the diff, every committed fixture derived from
  them, every research artifact downstream of a changed module, and every number recorded outside a
  driver artifact.
```

- [ ] **Step 3: Amend ADR-051 and fix the false CHANGELOG claim**

ADR-051:195 still reads *"`to_goal_relative_y`) deferred to PR 5 as a retrain trigger"* — the deferral
PR 5 closes. Add a section recording the closure, covering **both** geometry axes (the transform *and*
the grid re-anchor). Then `CHANGELOG.md:13` asserts *"No new ADR — §8b is recorded in ADR-051"*; §8b is
a section of the prior **spec**, not of the ADR. Pick one repair and make it unambiguous:
**recommended** — reword the CHANGELOG to cite the spec section and the new ADR-051 amendment by its
actual heading.

- [ ] **Step 4: Verify — falsifiably**

```bash
# The amendment must add a NEW heading recording the closure. Grepping for "chiral" or
# "to_goal_relative_y" CANNOT FAIL -- ADR-051:194-195 already contains both, in the deferral
# sentence being repaired. Rev 2 flagged exactly this shape in rev 1's "8b" check and then
# reintroduced it one line down. Anchor on a string that does not exist yet:
grep -n "^## .*Closure: PR 5" docs/superpowers/adrs/ADR-051*.md   # expect: exactly 1 hit
grep -n "deferred to PR 5" docs/superpowers/adrs/ADR-051*.md      # expect: reworded or marked CLOSED

# Untracked, EXCLUDING the spec and this plan (both legitimately untracked until Task 7):
git status --porcelain | grep "^??" | grep -v "docs/superpowers/" || echo "OK: no stray untracked"
```

The amendment must record **both** geometry axes, and one property in particular: the re-anchored
grid now **commutes with the ADR-028 point reflection** (old y-centres `1.5..67.5` are not closed under
`y -> 68-y`; new `1.0..67.0` are). That is *why* `space_controlled` became exactly chirality-invariant,
and it is a property a future edit could silently break.

**CORRECTED DURING EXECUTION — the commutation property IS guarded, by two existing tests.** This
step previously instructed the author to state that no gate covers it. That was false when written
and is independent of Gate 4's removal:

- `test_pr5_chirality_gates.py:151 test_grid_centres_are_mirror_symmetric` asserts
  `set(centres) == set(length - centres)` — the closure property itself, directly on the grid, and
  parametrized over both axes. Its own docstring records that the x half is already green and only y
  landed red, so "landed red" is not over-claimed.
- `:165 test_dominant_region_is_left_right_mirror_invariant` asserts the behavioural consequence —
  `space_controlled` invariant under a fixed-end left-right mirror.

So the ADR states the property AND names its guards. Recorded here rather than silently corrected,
per §11.4: a plan instructing a false statement is a plan defect, and this one survived seven review
rounds because every round reviewed the instruction, not the codebase it asserted about.

---

### Task 2: Provenance-wire `validate_xcross_causal`

**Files:** modify `scripts/validate_xcross_causal.py`, `tests/scripts/test_provenance_wiring.py`

**Interfaces:** consumes `scripts/_provenance.py` — `git_provenance() -> dict`,
`require_clean_tree(prov, *, allow_dirty: bool) -> None`. Produces `run_commit` / `run_tree_dirty` in
`docs/research/xcross_causal/metrics.json`.

- [ ] **Step 1: Enrol the driver and observe the gate RED — with the correct expectation**

```bash
python -m pytest tests/scripts/test_provenance_wiring.py -k xcross_causal -v > /tmp/pv.log 2>&1
grep -E "PASSED|FAILED|SKIPPED" /tmp/pv.log
```
There are **five** parametrized per-driver tests, not four. Expected on enrolment: **3 FAILED,
1 PASSED, 1 SKIPPED** — `test_driver_never_shells_out_to_rev_parse_directly` PASSES (it doesn't shell
out) and `test_the_guard_precedes_the_corpus_walk_within_main` SKIPS because `load_matches` is called
at `:166` inside `run()`, not in `main()`. **An agent told to expect four failures may "fix" a correct
driver.**

- [ ] **Step 2: Wire it, enforcing from `main()`**

Per ADR-037 the CLI refuses and `run()` records the truth — a `run()` that refuses on a dirty checkout
cannot be tested without mocking git.

```python
    parser.add_argument("--allow-dirty", action="store_true",
                        help="permit a dev run on a dirty tree; the artifact still records dirty=true")
    ...
    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)
    ...
    metrics["run_commit"] = prov["commit"]
    metrics["run_tree_dirty"] = prov["dirty"]
```

- [ ] **Step 3: Verify green**

```bash
python -m pytest tests/scripts/test_provenance_wiring.py -v > /tmp/pv.log 2>&1; grep -c FAILED /tmp/pv.log
```
Expected: `0`.

---

### Task 3: Write the covariate-invariance driver — write only, do not run

**Files:** create `scripts/measure_covariate_invariance.py`; modify `tests/scripts/test_provenance_wiring.py`

**Interfaces:** produces `docs/research/covariate_invariance/metrics.json` — a per-covariate table plus
`run_commit` / `run_tree_dirty` / `status`. **Run in Task 9.**

**Corpus:** the committed slim fixtures at `tests/datasets/tracking/action_context_slim/*_slim.parquet`
(sportec, skillcorner, metrica) — the same 804 frames §1.1's axis-A table used. **Record N in the
artifact**; a headline delta whose N is unstated is not a measurement.

- [ ] **Step 1: THREE arms, not two — the two-arm design cannot attribute `space_controlled`**

Commit 1 changed two things and they interact. Measured:

```
old y-grid 1.5..67.5 (centre 34.50):  reflect(1.5) = 66.5  -> NOT a grid centre -> not closed
new y-grid 1.0..67.0 (centre 34.00):  reflect(1.0) = 67.0  -> IS  a grid centre -> closed
```

The ADR-028 point reflection maps the **new** grid onto itself, forcing axis A to measure exactly zero
for `space_controlled` *by construction* — but the **baseline arm carries the old grid**, where the
reflection does not close, so axis A does move it there. Two arms leave the interaction unmeasured:

| | old grid | new grid |
|---|---|---|
| **old transform** | parent arm | axis-B leg |
| **new transform** | **the interaction — unmeasured by two arms** | current arm |

Run **three** arms: parent (old x old), current (new x new), and **new-transform x old-grid**. If the
third is dropped for cost, the artifact MUST label the `goal_x=105` residual **"A-plus-interaction"**,
never "A" — rev 2 asserted a clean two-axis decomposition that does not exist on this diff.

- [ ] **Step 2: Isolate the baseline by checkout, not by import**

```bash
mkdir -p "$HOME/pr5_scratch/baseline_tree"
git archive 6e3a132~1 | tar -x -C "$HOME/pr5_scratch/baseline_tree"
```

`git archive` rather than `git worktree` — worktrees are avoided in this workspace by standing
preference. (Either is mechanically safe: a worktree writes to `.git/worktrees/` and does not appear
in `git status --porcelain`. Preference, not correctness.)

**Why isolation at all, stated so it is not reverted:** both extractors bind geometry absolutely
(`_xcross_attempt.py:24`, `_xshot_occurrence.py:26`), so importing them under another package name
still resolves `_geo` to the *current* module. **That is inert for this diff** — commit 1's
`_geometry.py` change is purely additive and the baseline extractors contain zero references to
`to_goal_relative_y`. Two reasons to isolate anyway: the baseline arm would otherwise stamp the
*current* `GEOMETRY_VERSION`, and §4.5 bills this driver as reusable for PR 6, PR 7 and Cycle B, where
a `_geometry` function may change **behaviour** rather than being added — and it would then silently
measure zero, the exact silent-null shape CLAUDE.md catalogues four instances of. **Do not simplify
this on the grounds that today it makes no difference.**

- [ ] **Step 3: Reach every covariate — including the one that is not importable**

```python
from silly_kicks.causal.opportunities import (
    GK_BLOCK, PAPER_CONFOUNDERS, SHOT_ARM_CONFOUNDERS,
    LAYER2_BUILD_CONFOUNDERS, LAYER2_CONFOUNDERS, _COVARIATES,
)
_XS_GK_BLOCK = ("GK_r", "GK_theta")   # NOT importable -- literal gk_block= tuples at :139 and :198.
                                      # GK_BLOCK itself is the SIX lowercase xCross names.
# gk_depth_x is NOT in the union of the four confounder tuples (verified: 25 names, absent).
# It exists only as _COVARIATES["gk_depth_x"] (opportunities.py:395) and as
# layer2_config().treatment_covariate. Rev 2 asserted on it without enumerating it -> KeyError ->
# the driver would never have written the table, leaving 1.3's decision unbacked.
_TREATMENTS = dict(_COVARIATES)       # {"gk_depth_x": _covariate_depth}
```

Emit one row per covariate with `name`, `arm` (shot / cross / layer2-build / layer2-analysis /
treatment), delta at `goal_x=105`, delta at `goal_x=0`, and `axis` in {A, B, A+interaction, none}.
`theta` is at `opportunities.py:150`. Emit the **build-vs-analysis split** so Task 12 can cite this
artifact instead of re-deriving the stale-spells distinction in prose.

- [ ] **Step 4: Emit the four join-time confounders as data, not as an omission**

```python
_NOT_MEASURABLE = ("defensive_line_height", "defensive_line_compactness",
                   "pressure_on_actor__bekkers_pi", "time_remaining_s")
# Verified: none appears in XSHOT_ or XCROSS_FEATURE_NAMES_FAITHFUL; opportunities.py:152-157 says so
# ("per-spell joins, not extractor features"). Emit with
#   source="join-time (causal/_confounders.py)", delta="not-measurable-by-this-driver".
```

This matters beyond tidiness: §3.1 names `defensive_line_height` / `_compactness` as **PR 6's own
mechanism**, so an instrument billed as reusable for PR 6 cannot silently omit them. **Add the same
sentence to spec §4.5**, where the billing is.

- [ ] **Step 5: Positive control — the assertions must be able to FAIL**

Rev 2's only two assertions were `deltas["gk_depth_x"] == 0.0` and `deltas["GK_r"] == 0.0`. If the
baseline arm silently resolves to the current `_geo` — precisely the contamination Step 2 exists to
prevent — **every** delta is `0.0` and **both assertions pass**. That is CLAUDE.md's rule verbatim:
*"a gate that only asserts 'value is inside the band' passes identically when the computation silently
produced nothing."*

```python
# NEGATIVE controls (structural exactness -- what 1.3's decision rests on):
#   gk_depth_x @105 == 0.0   EXACT: cos is even
#   GK_r       @105 == 0.0   EXACT: hypot(a,-b) == hypot(a,b)
# POSITIVE controls (the computation actually happened) -- measured values available:
#   gk_theta          axis-A @105 ~= 2.65 rad
#   gk_lateral_offset axis-A @105 ~= 12.0 m
#   space_controlled  axis-B      ~= 26.6 m2 @105 / 35.5 m2 @0
assert any(abs(d) > 1e-6 for d in all_deltas), "all-zero table: baseline arm did not isolate"
assert baseline_geometry_version != current_geometry_version, "baseline resolved to current _geometry"
```

- [ ] **Step 6: Table first, status second — never a bare `assert`**

Task 8 Step 5 and §9 both say "record the verdict whichever way it lands." A hard `assert` does the
opposite: if an invariance genuinely breaks, the artifact documenting it is never written. Write the
table, set `status` in {`ok`, `invariance_breach`, `isolation_failed`}, then exit non-zero.

- [ ] **Step 7: Enrol in `ARTIFACT_DRIVERS`; expect 3 FAILED / 1 PASSED / 1 SKIPPED of five; wire; verify green.**

---

### Task 4: Write the platform-probe driver — write only, do not run

**Files:** create `scripts/measure_platform_probe.py`; modify `tests/scripts/test_provenance_wiring.py`

**Interfaces:** `--out DIR` emits `<platform>.json`, self-provenanced; `--compare A.json B.json` emits
the summary. **Both legs run in Task 9.**

- [ ] **Step 1: Cover BOTH extractors — an AND, not an OR**

Rev 2 permitted "either cover `extract_xcross_features` too or record the scope explicitly." The
permitted branch is wrong. The three bundled contracts hold 26 / 27 / 16 features, so §1.7's
`0.000e+00` covers **27 of 69**. Ghost and xShot are already empirically aarch64-clean — the xS probe
constructs both (`validate_xs_probe.py:141-142`) through `load()` -> `verify_feature_contract` and
completed 64/64 on the DGX. **Nothing in this plan loads `XCrossAttemptModel` on aarch64**
(`validate_xcross_causal.py:217` reads `metadata.json` directly and never constructs the model), while
`silly_kicks/tracking/features.py:780` loads it via `from_variant("default")` for
`xcross_attempt_xfns` — so any aarch64 consumer of that xfn hits the fail-closed prong. It is the one
contract-bearing artifact never loaded on aarch64, and it is reachable from a live public path. One
extra call, on machines Task 9 already visits.

- [ ] **Step 2: Emit what the hand-run JSON actually lacked**

Its top-level keys were `{features, platform, probe}` with `platform = {machine, python, system}` — so
"record platform identity" was already satisfied and was **not** the gap. The gap is **`run_commit`,
`run_tree_dirty`, `GEOMETRY_VERSION`, `probe_sha256`**. Emit all four.

- [ ] **Step 3: `--compare` must REFUSE a mismatched pair**

The legs run at different times on different machines and nothing structurally stops one being
launched from another commit. A delta between an x86 leg at commit 3 and a DGX leg at any other commit
**confounds platform with code** — the one thing this artifact exists to separate. Task 8 Step 1's
`git pull` makes agreement procedurally likely, not guaranteed. Refuse unless both legs agree on
`run_commit`, `GEOMETRY_VERSION` and probe identity, and both carry `run_tree_dirty == false`. Record
all four in the summary.

- [ ] **Step 4: Record the tolerance caveat rather than implying transfer**

`atol=1e-6` cannot transfer to the xCross vector even in principle: `space_controlled` is
`cell_count/805 x 7140`, quantized at **8.8696 m2 per cell, about 8.87e6 x atol**, so its
cross-platform error is exactly `0.0` or `>= 8.87` and the tolerance degenerates to an equality test.
`box_off_def_ratio` is likewise an integer ratio. An argmin flip is *unlikely* — 300 random 22-player
scenes gave a minimum relative first-vs-second gap of **6.80e-6**, none below 1e-12 — which is exactly
why skipping the measurement is indefensible: it is cheap, it will probably come back clean, and a
clean result is what PR 7 and ADR-050 need **on the record**.

- [ ] **Step 5: Do NOT carry the old README's verdict forward.** It states *"PR 7's ghost re-fit can
stamp on x86 and verify anywhere"* — derived from an xShot-only probe, asserted about the **ghost**
extractor, pre-empting what spec §6 makes PR 7's own acceptance item. Also state in the artifact that
the two legs confound architecture with interpreter (3.14.2/Windows/AMD64 vs 3.12.3/Linux/aarch64) and
that no third leg disentangles them, rather than leaving the `platform` fields to imply a decomposition
never made.

- [ ] **Step 6: Enrol in `ARTIFACT_DRIVERS`; land red; wire; verify green.**

---

### Task 5: Rebuild the fixture generator

**Files:** modify `scripts/make_xcross_directional_fixture.py`; regenerate
`tests/datasets/tracking/xcross_directional/frozen_rows.parquet`

**Rev 5 simplification — the fixture stays SINGLE-ENDED.** Rev 4 required both goal ends, reflection
pairs and a distinct-vector floor, all so a fourth gate could test chirality. That gate is dropped
(Task 6 Step 3): `tests/tracking/test_pr5_chirality_gates.py` already tests the point reflection over
**both** extractors (`:124-125`, parametrized 27 and 16 features), carries a **permanent** non-vacuity
plant (`:138`), covers **axis B** (`:165`), and extracts **live** — so it cannot be forged, which a
committed table can. Two ends bought nothing the stronger file did not already provide, and cost the
pairing requirement, the rows-vs-distinct-vectors confusion, and an unclosable forged-half hole.

Steps 2-4 below were **built and held** in review: 12 keeper positions with all six `gk_*` varying,
7 distinct `box_off_def_ratio` values, both `ten_minute_warning` states, and 24 negatives that stay in
domain.

- [ ] **Step 1: Add argparse — `--out` only**

Drop `--seed` and `--n-frames`. The generator is deterministic and its row count comes from the scene
lists, so `--seed` is a dead parameter and a *seeded* generator makes a committed golden
irreproducible. **Also expose an importable `build()` beside `main()`** — so the committed golden is
reproducibly regenerable, and so a future plant can regenerate in-process (measured 0.276 s);
`main()` is not callable without side effects. (The original reason — Task 6 Step 4's chiral
plant — no longer applies: that plant now lives permanently in `test_pr5_chirality_gates.py:138`
and none of the four remaining plants regenerates. Reason replaced rather than left standing, so
a later reader does not find a justification that fails and delete the export.)

- [ ] **Step 2: Vary the keeper** — `_build_frame:63` hard-pins
`_r(10, _DEF_TEAM_ID, 2.0, 34.0, is_gk=True)`. At least 8 positions, near/far post, advanced/deep.

- [ ] **Step 3: Vary `score_differential`, `ten_minute_warning`, `box_off_def_ratio`** — the clock
(`time_seconds >= 35*60` in period 1/2, `_xcross_attempt.py:258`) and the in-box ring occupancy.
`score_differential` is currently `np.nan` on every row (`:114`) and is the model's confounder #1.

- [ ] **Step 4: Negatives are wide + advanced but not cross-imminent** — slow ball, no arriving
teammate. Built and verified: 24 such scenes, all in `_in_wide_area`.

- [ ] **Step 5: Persist six columns** — `ball_x`, `ball_y`, `goal_x`, `gk_x`, `gk_y`, **`carrier_y`**.
Only three of the six GK features are computable without the carrier's `y`
(`gk_dist_near_post`, `gk_dist_far_post`, `gk_carrier_side` all need `side = sign(carrier_gr_y - 34)`).
Verified sufficient: re-deriving the full GK block from these columns matches the extractor to
**1.776e-15** across all rows. Note the re-derivation also pulls in the private
`_xcross_attempt._GOAL_HALF_WIDTH_M` and the `or 1.0` central-carrier guard — a train/serve
duplication that will drift, so keep it in one helper in the test file, not inline.

`main()` currently projects to `[*XCROSS_FEATURE_NAMES_FAITHFUL, "label"]`; widen it. The surviving
schema gate uses `issubset` and is unaffected.

- [ ] **Step 6: At least 40 rows AND 40 distinct feature vectors**

Single-ended, so the two are the same number and no reconciliation is needed. (Rev 4 asserted three
different floors — Gate 1's `len >= 40`, Step 8's `distinct >= 40`, and a narrative describing a
48-row build that Step 8 would have rejected. Dropping the pairing removes the discrepancy rather
than patching it.)

- [ ] **Step 7: Verify at the point of construction, INCLUDING the domain**

```bash
python - <<'EOF'
import pandas as pd
from silly_kicks.tracking._xcross_attempt import (
    XCROSS_FEATURE_NAMES_FAITHFUL as F, _in_wide_area, _ADVANCE_M)
df = pd.read_parquet('tests/datasets/tracking/xcross_directional/frozen_rows.parquet')
# NaN-EXPLICIT: `nan <= tol` is False, so a range clause ALONE misses an all-NaN column.
inert = [c for c in F if not df[c].notna().any()
         or float(df[c].max() - df[c].min()) <= 1e-9]
dom = [_in_wide_area(r.ball_x, r.ball_y, r.goal_x, _ADVANCE_M) for r in df.itertuples()]
distinct = len(df[list(F)].round(9).drop_duplicates())
print(f'rows={len(df)} distinct={distinct} inert={inert} in_domain={sum(dom)}/{len(df)}')
assert all(dom), 'OUT-OF-DOMAIN ROWS -- the precondition would block one task later'
assert not inert and distinct >= 40
assert df['label'].nunique() == 2 and (df['label'] == 0).sum() >= 3
assert {'ball_x','ball_y','goal_x','gk_x','gk_y','carrier_y'} <= set(df.columns)
EOF
```

---

### Task 6: Retire and replace the xCross liveness gate — THREE gates, plus a widening

**Files:** modify `tests/tracking/test_xcross_attempt_integration.py:415-424` and
`tests/tracking/test_pr5_chirality_gates.py`

- [ ] **Step 1: MEASURE the GK response on the NEW fixture, at GOAL-RELATIVE probes**

Non-negotiable ordering (§4.6). Pin probes in **goal-relative** coordinates: `gk_x`/`gk_y` are
absolute, so a pinned absolute `(2.0, 34.0)` is a keeper on his line at one end and 103 m upfield at
the other — the gate still passes, so it would fail *quietly*.

**Use `mean(|dp|)`, never `|mean dp|`.** Measured over 7 goal-relative probes: `mean|dp|` spread
**1.50x** (0.00333-0.00501) versus signed-mean **139x** (3.58e-05-0.00499). Base rate 0.012217, so
relative `mean|dp|` is **27.29-41.01%** and a ~0.15 relative threshold carries 1.8x headroom. Derive
the constant here; do not transcribe that one either — it is fixture-specific.

- [ ] **Step 2: Write THREE gates, with docstrings that claim only what they test**

```python
def test_xcross_fixture_is_not_degenerate():
    """PRECONDITION. Lands RED on the old fixture -- it fails all three clauses."""
    df = pd.read_parquet(_XCROSS_DIRECTIONAL)
    assert len(df) >= 40
    # NaN-EXPLICIT and RANGE-based. Measured on the committed 16-row fixture: range alone finds 8
    # (misses all-NaN score_differential, since `nan <= tol` is False), nunique alone finds 7
    # (misses gk_dist_near_post/_far_post, which differ by 3.0 ULP), this clause finds all 9.
    # _MIN_RANGE holds FROZEN LITERALS measured at authoring time. It must NOT be "a fixed
    # fraction of the observed range": measured, frac=0.001 and frac=0.01 each find only 7 of 9,
    # missing exactly the two ULP-separated columns this clause exists for -- and computed from
    # the dataframe at test time it is degenerate (`range <= frac*range` is False for every
    # frac < 1, True for all 16 at frac == 1.0).
    inert = [c for c in XCROSS_FEATURE_NAMES_FAITHFUL
             if not df[c].notna().any()
             or float(df[c].max() - df[c].min()) <= _MIN_RANGE[c]]
    assert not inert, f"inert features cannot contribute to any ranking: {inert}"
    assert _all_rows_in_wide_area(df)

def test_xcross_bundled_model_predictions_are_live():
    """LIVENESS -- finite, in [0,1], non-constant.

    RENAMED from test_xcross_bundled_model_is_live_not_degenerate. Reusing the retired gate's
    exact name with a different body hides the change from git history and from any
    name-anchored guard.
    """

def test_xcross_bundled_model_reads_the_gk_block():
    """RESPONSE. Claims what it tests: perturbing the GK feature BLOCK moves the prediction.

    NOT 'responds to keeper position' -- a real keeper move also shifts dist_nearest_def (up to
    5.35 m) and space_controlled (up to 133.04 m2), which a block-only sweep holds fixed. The
    swept state is not a realizable frame; it is a probe of which columns the model reads.
    mean(|dp|) over the block, relative to the fixture's own base rate, at pinned goal-relative
    probes. Direction NOT asserted.
    """
```

- [ ] **Step 3: DROP the fourth gate; widen `pr5_scene()` instead**

Rev 4 added a chirality mirror-pair gate here. It is **duplicative and strictly weaker** than
`tests/tracking/test_pr5_chirality_gates.py`, which already covers both extractors (`:27-28`,
`:124-125`), both axes (`:165` is the axis-B sibling — it exists; do not add another), and carries its
non-vacuity plant **permanently** (`:138`) rather than as a one-off manual step. Above all it extracts
**live**, so it cannot be forged, whereas a gate reading a committed table cannot distinguish an
honestly-extracted fixture from a fabricated one **at any tolerance**: measured, a forged half is
bit-identical to a real extraction on integer coordinates, and the float-noise discriminator
`(deltas != 0).any()` is brittle — `100.7` round-trips exactly through `105 - (105 - v)` while `4.8`
does not, so an author changing the carrier offset from `+0.3` to `+0.5` would get a genuine fixture
flagged as forged.

**Marginal value of a fourth gate would have been "48 scenes instead of 1, one extractor, one axis."**
Take that benefit where it is strongest: widen `pr5_scene()` to more scenes in the file that already
extracts live and covers both extractors and both axes.

- [ ] **Step 4: Plant FOUR defects**

| plant | must fail | note |
|---|---|---|
| constant model | liveness, response | |
| **GK-blind but live** model | response only | measured AUC **1.0000**, *above* the real model's 0.9913, at **0.00%** response — survives liveness AND ranking |
| GK-pinned fixture | precondition **and response** | measured `G1 FAIL / G3 FAIL`: pinning the keeper makes the GK block inert, which trips the precondition AND leaves the response gate nothing to move. Listed in full because Step 4 says record each observed-failing — a recorder who sees two gates where the table predicts one is the discrepancy the out-of-domain row above was corrected to remove |
| out-of-domain row | precondition | |

Do **not** transcribe the chiral-plant constants from review; they are fixture-specific (22.0 / 2.929 /
2.871 on one build versus 21.0 / 2.559 / 2.871 on another). That plant now lives permanently in
`test_pr5_chirality_gates.py:138` and needs no manual step here.

- [ ] **Step 5: Decide the ranking axis explicitly**

The retired AUC gate was vacuous **because the fixture was degenerate**; once the precondition holds it
is not (measured AUC 0.9913 on the repaired fixture). But a retained `AUC >= 0.9` is **green on the
GK-blind plant at 1.0000** — so if ranking is kept, the comment must say it is not a substitute for
the response gate. ADR-032 exists so a later reader meets the reasoning rather than a hole.

- [ ] **Step 6: Delete the retired gate with the reason in place.**

---

### Task 7: Gates, `/final-review`, COMMIT 3, push

- [ ] **Step 1: Full suite**

```bash
python -m pytest tests/ -m "not e2e" -q > /tmp/suite.log 2>&1; echo "exit=$?"
grep -E "^(FAILED|ERROR)" /tmp/suite.log; tail -3 /tmp/suite.log
```
Expected: 0 failed.

- [ ] **Step 2: Lint and types at CI scope**

```bash
python -m ruff check silly_kicks/ tests/ scripts/ && \
python -m ruff format --check silly_kicks/ tests/ scripts/ && python -m pyright
```

- [ ] **Step 3: Ask the §8 item-14 question** (see Task 14 — it is commit-3-timed because a
"provisional" answer edits `train_gk_completion.py`, which is commit-3 code).

- [ ] **Step 4: `/final-review`**, fix findings in the working tree.

- [ ] **Step 5: Set the five version files aside — they are commit 4's**

```bash
git stash push -m "commit4-version" -- \
  pyproject.toml silly_kicks/__init__.py uv.lock TODO.md CHANGELOG.md
git status --porcelain    # inspect BEFORE staging
```
**Rev 1 ran `git add -A` here.** That silently staged the `4.73.0 → 4.74.0` bump plus the whole
`## [4.74.0]` CHANGELOG section into commit 3 — violating the plan's own File-structure table, spec
§4.0, the "no version number until commit-prep" constraint, and falsifying Fact 2, whose entire point
is that commit 3 contains no `silly_kicks/` change. (CHANGELOG is stashed too, then restored in Task
15; if Task 1 Step 3's ADR fix touched it, re-apply that edit after the stash pop.)

- [ ] **Step 6: Stage path-scoped and verify BEFORE committing**

```bash
git add CLAUDE.md docs/ scripts/ tests/
git status --porcelain                       # expect: staged entries only, no '??', no version files
git diff --cached --name-only | grep -E "^(pyproject|uv.lock|silly_kicks/__init__)" && \
  echo "STOP: version files staged" || echo "OK"
```
Rev 1's check was `git add -A && git status --porcelain` "expect zero `??`" — which **cannot fail**,
since `add -A` converts every `??` to `A ` by definition. It established nothing and hid the defect
above.

- [ ] **Step 7: Commit and push**

```bash
git commit -F "$HOME/pr5_scratch/commit3_msg.txt"
git push origin adr051-pr5-chiral-transform
```

- [ ] **Step 8: Confirm the tree is clean for the driver half**

```bash
git stash pop && git status --porcelain   # expect: ONLY the five version files
```
The five version files are the sole permitted dirt entering commit 4 — and they are **not** clean for
a driver run. **Restash them for the duration of Task 8–11** and pop at Task 15:

```bash
git stash push -m "commit4-version" -- pyproject.toml silly_kicks/__init__.py uv.lock TODO.md CHANGELOG.md
git status --porcelain   # expect: EMPTY. Drivers can now run.
```

---

# COMMIT 4 — every driver first, one copy-in, then prose

**Structural rule for this half:** every driver writes to `$HOME/pr5_runs/<name>`. Nothing is copied
into `docs/research/` until Task 11, and no prose is edited until Task 12. Rev 1 interleaved driver
runs with in-repo writes, so whichever ran second hit `SystemExit`.

### Task 8: DGX — the two causal re-runs

- [ ] **Step 1: Sync the DGX and confirm clean**

```bash
ssh karsten@192.168.68.73 'cd ~/Development/silly-kicks && git fetch origin && \
  git checkout adr051-pr5-chiral-transform && git pull --ff-only && \
  git rev-parse --short HEAD && git status --porcelain'
```
Expected: the commit-3 SHA and **empty** status. The DGX was verified clean at `08ce9a8` with the xS
probe already complete and its output outside the repo, so the pull is safe. **If status is non-empty,
stop and find out why — do not pass `--allow-dirty`.**

- [ ] **Step 2: `tf19_entanglement` re-run** — `python scripts/validate_xshot_causal.py --out ~/pr5_runs/entanglement`. 179 matches / 98,789
opportunities; `for_each`-sharded and resumable. Background; poll no faster than the shard cadence.

- [ ] **Step 3: Re-assert cleanliness, then `xcross_causal`** — `python scripts/validate_xcross_causal.py --out ~/pr5_runs/xcross_causal`.
23,966 opportunities; first provenanced version. Its GK block is the **six lowercase names** and its
confounders are `PAPER_CONFOUNDERS`, which include `space_controlled` — exposed to **both** axes.

- [ ] **Step 4: If the entanglement token moved, recompute — do NOT re-run the probe.**
`regate_verdict(*, arm, probe_verdict, entanglement)` (`_model_eval.py:714`) is pure over three string
tokens.

- [ ] **Step 5: Record the verdict whichever way it lands.** A flip is a result, not a failure.

---

### Task 9: Run the remaining drivers — all out-of-repo

- [ ] **Step 1: Platform probe, DGX leg** — `--out ~/pr5_runs/platform_probe`; scp the JSON back.
- [ ] **Step 2: Platform probe, x86 leg** — `--out "$HOME/pr5_runs/platform_probe"`.
- [ ] **Step 3: `--compare` the two legs**, output to `$HOME/pr5_runs/platform_probe/`.
- [ ] **Step 4: Covariate-invariance driver** — `--out "$HOME/pr5_runs/covariate_invariance"`, using
the Task 3 Step 1 worktree as `--baseline-tree`.

**Expected new information:** the cross-arm columns §1.1's table never covered — `space_controlled`
under axis B, `gk_lateral_offset` under axis A. Record what comes out; §1.1's ⛔ is discharged here.

- [ ] **Step 5: Confirm the repo is still clean**

```bash
git status --porcelain   # expect: EMPTY -- every output went to $HOME/pr5_runs
```

---

### Task 10: Collect the xS probe output

- [ ] **Step 1: Verify the non-dependency over ALL of `silly_kicks/`**

```bash
git diff --name-only 08ce9a8..HEAD -- silly_kicks/
```
Expected: **empty**. Rev 1 grepped three paths, which would stay empty even if the version bump had
landed in commit 3 — and the probe also reaches `_xcross_attempt.py`, `_ball_carrier.py`,
`pitch_control/` and `_das.py` through the gkdv engine. This is the check Fact 2 actually asserts.
**If non-empty, classify every hit before trusting the probe output.**

- [ ] **Step 2: scp the probe artifact, and SPLIT it explicitly**

`validate_xs_probe.py` takes **one** `--out` and runs both variants into it
(`:140  variants = ["v1", "v2"] if variant == "both" else [variant]`), but Task 11 copies into **two**
research directories, which are not parallel in shape today — `tf19_pr3b` holds `metrics.json` +
`report.md`; `tf19_pr3b_xs_v2` holds a README, two metrics files and two reports. They were
historically produced by separate runs.

**State the mapping, because a claim depends on it.** Task 10 asserts `tf19_pr3b`'s missing provenance
"self-heals because the probe ran `--variant both`" — verified that its `metrics.json` carries
`run_commit` **ABSENT** today, so the heal is real only if the v1 leg actually lands there. Write down
which keys of the probe output become which file in which directory, or K3 stays open while the plan
believes it closed.

The artifact carries `run_commit: 08ce9a8`, an ancestor of the tip — **correct, and not to be "fixed"
to the tip.**

---

### Task 11: One copy-in step

- [ ] **Step 1: Copy every output into `docs/research/`** — `tf19_entanglement`, `xcross_causal`,
`tf19_pr3b`, `tf19_pr3b_xs_v2`, `pr5_platform_atol`, `covariate_invariance`. This is the **only** step
that writes into the repo before the prose edits, and it runs after every driver has finished.

- [ ] **Step 2: Confirm each carries provenance — by GLOB, not a hardcoded filename**

Rev 3 hardcoded `metrics.json` for all six. `pr5_platform_atol/` holds `README.md`, `dgx.json`,
`x86.json` and **no `metrics.json`** (verified), so the single check certifying the whole copy-in
prints `ABSENT` for a *correctly provenanced* directory. The executor then either "fixes" a correct
artifact or learns that ABSENT is normal — disarming the check for the five where ABSENT would be
real.

```bash
python - <<'EOF'
import json, pathlib
dirs = ["tf19_entanglement", "xcross_causal", "tf19_pr3b", "tf19_pr3b_xs_v2",
        "pr5_platform_atol", "covariate_invariance"]
bad = []
for d in dirs:
    js = sorted(pathlib.Path("docs/research", d).glob("*.json"))
    assert js, f"{d}: no JSON at all"
    for f in js:
        m = json.loads(f.read_text(encoding="utf-8"))
        ok = m.get("run_commit") and m.get("run_tree_dirty") is False
        print(f"  {d}/{f.name}: run_commit={str(m.get('run_commit'))[:12]} dirty={m.get('run_tree_dirty')}")
        if not ok:
            bad.append(f"{d}/{f.name}")
assert not bad, f"unprovenanced: {bad}"
EOF
```

This also covers the `--compare` summary, whose filename Task 9 Step 3 never states.

---

### Task 12: Prose — `tf19_signoff_power`, `tf19_pr2`, CHANGELOG

- [ ] **Step 1: Annotate `tf19_signoff_power` — THREE classes, in a SIBLING file**

Rev 3 said "verdict-bearing vs knowingly-stale". That binary cannot express what the artifact actually
holds:

| class | keys |
|---|---|
| **Invariant** (treatment-derived, §1.3) | `n_spells` 37086 · `n_treated` 151 · `treated_prevalence` 0.004072 · `sizes` · `N_MIN_MATCHED` · `n_min_per_outcome` |
| **Stale now** (PR 5 moved `theta`) | `att` — both outcomes |
| **Current now, stale after PR 7** | `icc` — the leg that discharged ADR-037 F2 |

A two-way annotation asserts `icc` is current, and it will be wrong two PRs later with no mechanism to
revisit. Annotate **per leg**, with a `pending_invalidation` field naming PR 7 on the `icc` leg. The
artifact already distinguishes them — `upstream_provenance.spells` and `upstream_provenance.arm_values`
carry *different* `run_commit` values — so the split is available, not invented.

**Write it to a SIBLING `invalidation.json`, never into `metrics.json`.** Hand-editing a
driver-produced file under an unchanged `run_commit` is the mirror image of the restamp §4.5 forbids:
adding content the recorded commit did not produce. The sibling cites the metrics artifact and
`covariate_invariance` by path, leaves the driver output byte-identical, and makes §9's "marks its
stale fields" checkable instead of unfalsifiable.

Also state why re-running only the analysis leg would be wrong: `theta` is in
`LAYER2_BUILD_CONFOUNDERS` (`opportunities.py:150`), a **per-spell stored column**, so the persisted
spells parquet is stale in `theta` and a seconds-long analysis re-run would **launder** the decimals
rather than fix them. **Do NOT restamp `run_commit`** — it stays `6b242cfb…`.

- [ ] **Step 2: Rewrite `tf19_pr2/decision_table.md` — and provenance BOTH legs**

The *after* leg is `_xcross_weights/default/metrics.json` (`run_commit: 6e3a132`, `ratio 1.7013`). The
*before* leg (`ratio 1.4076`, `gk_median_abs_delta 0.002417`) lives only at
`git show 6e3a132:silly_kicks/tracking/_xcross_weights/default/metrics.json` and carries **no**
provenance — it predates the wiring. §9 requires every cited number to be backed by a driver-produced
artifact, so either cite the before leg as `git show <sha>:<path>` **and state plainly that it predates
provenance wiring**, or drop the before/after framing and record only the current provenanced value.
Rev 1 fixed the citation and left the acceptance criterion failing on the same line.

**Take option (a). The two are not equivalent.** Dropping the before/after framing discards the
CHANGELOG's actual claim — *"TF-19 verdicts re-run on the corrected weights, both UNCHANGED"* — and
that invariance is the substantive result, not decoration. Cite the before leg as
`git show 6e3a132:<path>`, and state in the citation that **a `git show` SHA anchors when the file was
committed, not what code produced it** — precisely the distinction `run_commit` exists to make, and
the reason this leg is weaker evidence than the after leg rather than equal to it.

- [ ] **Step 3: Disambiguate the variant.** The bundled `public` default and Stage-B `sc_extended`
report different ratios; the current text does not say which is which.

---

### Task 13: HF uploads and model cards (items 3 and 20)

- [ ] **Step 1: Re-read `docs/research/tf19_pr2/hf_upload_instructions.md` FIRST** (§10 K5) — it states
each `*-v1` repo serves exactly one Hub variant at its root. Confirm against the upload plan **before**
uploading.

- [ ] **Step 2: Upload the xShot `sc_extended` artifact.** Derived parameters are shareable; raw
provider data is not. The test is reversibility, not corpus provenance.

- [ ] **Step 3: Upload the retrained xCross `sc_extended` — the "or record it" branch is closed**

`load()`'s `geometry_version` prong only warns, but the feature-contract prong is fail-closed and the
probe frame is unchanged, so the pre-PR-5 Hub artifact takes the fingerprint-comparison branch and
`XCrossAttemptModel.from_hub()` (`_xcross_attempt.py:673`) **raises `IntegrityError`** on a public
path. Rev 3 offered "upload **or** record it as knowingly unloadable" and then closed with *"Public
API silently broken is not an option"* — the two branches are not both acceptable under that sentence.
Branch (b) is shipping it broken, documented rather than silent. **Upload it.**

**State the Hub revision strategy before uploading.** Step 1's own premise is that each `*-v1` repo
serves exactly one variant at its root, so an upload **replaces** — an outward-facing, hard-to-reverse
action on a public repo. Decide and record: overwrite at root, or push to a revision/tag and move the
pointer. This is a publish, not a file copy.

- [ ] **Step 4: Update both model cards** under `docs/huggingface/model-cards/` — they document the
chirality/geometry stamps. (`ghost-gk-v1` is PR 7's, per §6.)

---

### Task 14: The item-14 question — asked at Task 7, gated at the next trainer run

**This is NOT a tag gate.** Measured: **zero** weight artifacts carry `artifact_label`/`all_public`,
and `_corpus_taxonomy` exists only in `train_gk_completion.py`, which PR 5 never runs. PR 5 ships the
**code path**, not the label; the commit message's *"will now be labelled `full`"* is forward-looking.
Rev 1 (following spec rev 3) blocked commit 4 and the tag on an owner round trip for a label the PR
does not emit.

- [ ] **Step 1: Ask at Task 7 Step 3** — before commit 3 is pushed, because a "provisional" answer is
an edit to `train_gk_completion.py`, which is commit-3 code.
- [ ] **Step 2: Register the real trigger (item 22)** — the next `train_gk_completion` run: PR 7,
TF-24, or any re-materialization. Nothing carries a gate there today.
- [ ] **Step 3: Record the residual, not as a gate** — `train_xcross_attempt.py:623` uses
`artifact_label` for **variant selection**, so the taxonomy rule may have influenced which weights PR 5
bundles even though no label string was written. That is item 15's territory.

---

### Task 15: Commit-prep, gates, COMMIT 4, PR

- [ ] **Step 1: Restore the version files and merge `origin/main` FIRST**

```bash
git stash pop                      # the five files from Task 7 Step 8
git fetch origin && git merge origin/main
```
Only then assign version / PR-S / ADR from the merged state. Five collisions in this cycle.

- [ ] **Step 2: Write the version to all five sites** — `pyproject.toml`, `silly_kicks/__init__.py`,
`uv.lock` (via `uv lock`), CHANGELOG heading, TODO "Current release". They are already at `4.74.0` in
the stash; re-check against the merged state rather than assuming.

- [ ] **Step 3: Groom TODO.md** — delete shipped rows, do not annotate them.

- [ ] **Step 4: Full gates** (Task 7 Steps 1–2) and `/final-review`.

- [ ] **Step 5: Acceptance check against §9, item by item** — every re-run directory carries a
`run_commit` it earned; **no number cited in CHANGELOG or a research directory unbacked by a
driver-produced artifact** (including Task 12 Step 2's before leg); `tf19_signoff_power` marks its
stale fields; all three bundled models load under both chirality and feature-contract enforcement **AND** every published `sc_extended` variant loads via `from_hub` — `from_variant` is NOT `from_hub`, and rev 3's criterion would go green on a release where `XCrossAttemptModel.from_hub()` raises `IntegrityError` on a public path;
C4 count unchanged.

- [ ] **Step 6: Present the diff, get explicit approval, COMMIT 4.**

- [ ] **Step 7: Open the PR requesting `--merge`, NOT squash.** The repo default is squash-only; a
squash rewrites `6e3a132` and `08ce9a8` and orphans the `run_commit` citations in four wheel artifacts
and six research directories.

---

## Self-review

**Spec coverage.** §4.0 → Tasks 1, 7, 15. §4.1 → Task 10. §4.2 → Task 2. §4.3/§4.4 → Task 8. §4.5 →
Tasks 3, 9, 12. §4.6 → Tasks 5, 6. §4.7 → Task 13. §4.8 → Task 1. §4.9 → Task 15. §4.10 → Tasks 4, 9.
§8/§9 item-14 → Tasks 7 Step 3, 14. Items 16–20 → Tasks 5, 12, 4+9, 1, 13. Item 22 → Task 14 Step 2.
Items 21 and Cycle B are out of this plan by §3.

**The one deliberate blank.** Task 6 Step 2's response threshold cannot be written until Step 1
measures it on the new fixture — that is the ordering constraint, not an omission. The two surviving
shape constraints (joint, non-monotone) are fixed in advance; the third is now stated as a property of
the assertion form rather than as a number that rots when the fixture's base rate changes.

**What earlier revisions got wrong, so none of it is reintroduced.** Rev 1 ran guarded drivers in
the commit-3 half and interleaved runs with in-repo writes — both `SystemExit`. Rev 2 fixed the
sequencing and left the driver *contents* unexamined: a two-arm design that cannot attribute the one
covariate both axes touch, an assertion set satisfied by the contamination its own isolation step
exists to prevent, a scope `OR` that would have left the only aarch64-unverified contract unmeasured,
a response gate using the signed-mean form (measured to span 67x by probe position), a precondition
the specified fixture cannot satisfy, an in-domain restriction that destroys the negative class, and
a regression fixture built at a single goal end — in the PR that closes the chirality defect class.

**Not in this plan, by design.** Cycle B (§5), PR 6, PR 7, TF-24, item 21, and the two §8
re-registrations.
