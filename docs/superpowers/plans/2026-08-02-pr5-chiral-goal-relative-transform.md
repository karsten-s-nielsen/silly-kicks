# PR 5 — Chiral Goal-Relative Transform + xS/xCross Retrain — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax. Execution is INLINE by owner
> preference — do NOT dispatch subagents.

**Goal:** Make the goal-relative transform a 180-degree point reflection so one physical scene scores
identically at both goal ends, and retrain the two models whose fail-closed stamps that invalidates.

**Architecture:** Add `to_goal_relative_y`/`_vy`; apply the reflection once per extractor (one seam in
xCross, a whole-frame pre-transform in xShot); re-anchor the dominant-region grid symmetrically; bump
`GEOMETRY_VERSION`; retrain both models on both arms and re-stamp. Detection lands before the fix.

**Tech Stack:** Python 3.10–3.14, numpy, pandas, xgboost, pytest. DGX Spark (aarch64) for fits, x86
for stamping.

**Spec:** `docs/superpowers/specs/2026-08-01-pr5-chiral-goal-relative-transform-design.md`. §N refers
there. Read §0 — it records what three review passes changed.

**Revision note (2026-08-02, reviews #3–#5):** restructured from two commits to three; every gate
snippet rewritten (the previous versions could not run, and their `TypeError` would have produced
exactly the "2 FAILED" the plan named as the expected red); the cp1252 gate's population narrowed to
scripts that actually have a parser, after it was found it would have executed 16 that rewrite
committed goldens and checksums; and the platform-probe check made elementwise against the stored
fingerprint.

## Global Constraints

- **THREE commits on ONE feature branch (`adr051-pr5-chiral-transform`). Never pushes to `main`.**
  Commit 1 = code + spec + plan. Commit 2 = weights + stamps. Commit 3 = research dirs + docs.
  No WIP commits, nothing staged between, no squash-later.
- **The spec and this plan are staged WITH commit 1** — never as a commit of their own. This is
  required, not stylistic: `scripts/_provenance.py:73` counts UNTRACKED files as dirty **on purpose**,
  and every artifact driver calls `require_clean_tree` from `main()`. Left uncommitted they make
  `git status --porcelain` dirty and **every fit and probe refuses**.
- **Merge with `--merge`, NEVER squash.** Commit 1's SHA is stamped into four wheel artifacts and
  commit 2's into six research directories; a squash rewrites both and orphans every citation. The
  repo default is squash-only, so this is an explicit request.
- **All fit outputs go OUTSIDE the repo** (`$HOME/runs/pr5/...`). An `--output-dir` inside the repo
  creates `?? runs/` and the *next* `require_clean_tree` refuses.
- **No version number until commit-prep**, and only after `git fetch && git merge origin/main`.
- **Lint/type — use the EXACT CI scope**, not `.`:
  `python -m ruff check silly_kicks/ tests/ scripts/`,
  `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright` (bare;
  config-driven include). Neither tool is on PATH. Measured during execution: `ruff check .` walks
  `.venv/` and `calibration_runs/` and reports **234 pre-existing errors** that are not the repo's
  and not yours — enough noise to hide the 2 real ones.
- **Never read a gate verdict through `| tail`** — redirect to a file and grep by name.
- **`/final-review` runs before each of the three commits**; findings fixed in the working tree.
- **Never assume a CLI flag exists.** Read `--help` first. Reviews #3 and #4 found four invented
  flags between them — and the fourth is what motivates the rule immediately below, because no
  amount of `--help`-reading would have caught it.
- **NEVER invoke a `scripts/*.py` that has no argument parser — not even with `--help`.** Check
  `grep -q add_argument` first. Without a parser, `--help` is not rejected: it is **ignored** and
  `main()` runs. Sixteen public scripts are in that class and they rewrite committed baselines,
  goldens and `SHA256SUMS`, or pull over the network. This rule is strictly stronger than the one
  above, and it is the one that matters: the "read `--help` first" rule **cannot** catch it, because
  there is no parser to reject the flag. Applies to gates, discovery loops and ad-hoc probing alike.
- Constants: `FIELD_LENGTH = 105.0`, `GOAL_Y = 34.0`, `PITCH_LENGTH = FIELD_LENGTH`,
  `PITCH_WIDTH = GOAL_Y * 2.0`. There is **no** `FIELD_WIDTH`.

## Verified API facts (review #3 — do not re-derive, do not assume otherwise)

```python
# tests/tracking/_mirror_registry.py:125  -- NOT conftest
def canonical_scene() -> tuple[pd.DataFrame, pd.DataFrame]:   # (actions, frames)

# silly_kicks/tracking/_xshot_occurrence.py:150 -- returns a 1-ROW DataFrame
def extract_xshot_features(frame_data, *, gk_team_id, goal_x, feature_set="faithful") -> pd.DataFrame

# silly_kicks/tracking/_xcross_attempt.py:126 -- carrier_player_id is REQUIRED
def extract_xcross_features(frame_data, *, gk_team_id, goal_x, carrier_player_id,
                            feature_set="faithful", score_differential=np.nan) -> pd.DataFrame

# scripts/_corpus.py:32 -- keyword-only ARRAYS, returns a per-row MASK
def is_public_row(*, providers: np.ndarray, match_ids: np.ndarray,
                  visibility: dict[tuple[str, str], str]) -> np.ndarray
```

The extractors take **one frame snapshot**. `canonical_scene()`'s `frames` holds 3 frames x 23 rows;
passing all 69 does **not** raise — it silently scores the wrong row.

---

## PHASE A — code (Commit 1)

### Task 1: Thread `cache_dir` through the trainers

**Files:** `scripts/train_xshot_occurrence.py` (`:78`, argparse), `scripts/train_xcross_attempt.py`
(`:70`, argparse), `scripts/train_gk_completion.py` (`:71`, argparse)

**NOT `train_gk_retention.py`** — verified it does not call `load_matches`.

- [ ] **Step 1: Confirm the three call sites**

Run: `grep -n "load_matches(" scripts/train_*.py`
Expected: exactly `train_xshot_occurrence.py:78`, `train_xcross_attempt.py:70`,
`train_gk_completion.py:71`, none passing `cache_dir`.

- [ ] **Step 2: Add the flag to each of the three**

```python
parser.add_argument(
    "--cache-dir",
    type=Path,
    default=None,
    help=(
        "Persist downloaded pining artifacts under CACHE_DIR/{provider}/{match_id}/ and reuse them "
        "on later runs over the same corpus. Default None re-downloads every run. The cache is "
        "keyed on (provider, match_id) only, so it serves stale bytes if an upstream artifact is "
        "revised; these are immutable historical matches."
    ),
)
```

- [ ] **Step 3: Pass it through at each call site**

```python
yield from load_matches(
    providers=providers, match_ids=match_ids,
    max_per_provider=max_per_provider, cache_dir=cache_dir,
)
```

Default `None` must stay byte-identical to today.

- [ ] **Step 4: Verify**

Run: `python scripts/train_xshot_occurrence.py --help > /tmp/pr5/t1.txt 2>&1; grep -c "cache-dir" /tmp/pr5/t1.txt`
Expected: `1`. Repeat for the other two.

---

### Task 2: Wire ADR-038 taxonomy into `train_gk_completion.py`

**Owner decision, 2026-08-02: wire it in and ship the `"full"` label.** No longer blocked.

Context so the next reader does not re-open it: `--providers` defaults to `["gradientsports"]`
(`:581`), and `artifact_label` maps any non-all-public GS run to **`"full"`** — the most restricted
tier — so the emitted `metrics.json`, which ships inside the PyPI wheel, will newly carry that label
where today it carries none. That was accepted on the grounds that those GS-derived coefficients
**already ship**: the label documents an existing situation rather than creating one, and
`assert_public_corpus` only gates the `public` claim, so nothing breaks.

Note the two risks are different and only the first is what L26 names: the **SkillCorner** gap (a
defaulted `--max-per-provider 64` run pulling 54 restricted matches into a wheel with nothing
refusing it) is what the guard fixes; the **Gradient Sports** `"full"` label is a side effect of the
same change.

- [ ] **Step 1: Record the pre-change state**

```bash
for v in default skillcorner; do echo "--- $v"; \
  python -c "import json;print(json.load(open('silly_kicks/tracking/_gk_completion_weights/$v/metrics.json')))"; done
```
Expected: `default` has `providers=['gradientsports']`, no `artifact_label`; `skillcorner` has
`n_matches=10`, `bundled=True`, `run_commit=4b15365…`, `run_tree_dirty=False`.

- [ ] **Step 2: Copy the shape the xS/xCross trainers already use — do not invent one**

Read `scripts/train_xshot_occurrence.py:442-452` and mirror it. `is_public_row` takes keyword-only
**arrays** and returns a per-row **mask**:

```python
import numpy as np
from scripts._corpus import artifact_label, assert_public_corpus, is_public_row

providers_arr = np.asarray([p for p, _ in loaded_keys])
match_ids_arr = np.asarray([m for _, m in loaded_keys])
mask = is_public_row(providers=providers_arr, match_ids=match_ids_arr, visibility=visibility)
all_public = bool(mask.all())
if all_public:
    assert_public_corpus(visibility)
metrics["artifact_label"] = artifact_label(
    providers=set(providers_arr.tolist()), all_public=all_public
)
```

- [ ] **Step 3: Unit-test the fail-closed direction (no CLI run — there is no `--dry-run`)**

`train_gk_completion.py` has **zero** `dry-run` occurrences. Test the helpers directly:

```python
def test_gk_completion_label_is_restricted_when_any_row_is():
    vis = {("skillcorner", "1"): "public", ("skillcorner", "2"): "private"}
    mask = is_public_row(
        providers=np.array(["skillcorner", "skillcorner"]),
        match_ids=np.array(["1", "2"]), visibility=vis,
    )
    assert mask.tolist() == [True, False]
    assert artifact_label(providers={"skillcorner"}, all_public=False) == "sc_extended"
    assert artifact_label(providers={"skillcorner"}, all_public=True) == "public"
    assert artifact_label(providers={"gradientsports"}, all_public=False) == "full"
```

Run: `python -m pytest tests/ -k gk_completion_label -q`
Expected: PASS.

---

### Task 3: Fix the cp1252 `--help` crash

**Files:** the crashing `scripts/*.py` drivers; create `tests/scripts/test_help_is_cp1252_safe.py`

> ### DANGER — read before running anything in this task
>
> **A script with no argument parser does not reject `--help`. It ignores it and runs `main()`.**
> Measured: **16 public `scripts/*.py` contain no `argparse`/`add_argument` at all** —
> `stamp_feature_contracts.py` (rewrites `metadata.json` + `SHA256SUMS` for all three bundled
> artifacts), `regenerate_action_context_baselines.py`,
> `regenerate_pressure_snapshot_shas.py`, `regenerate_provider_defaults.py`,
> `gen_ghost_gk_kde_golden.py`, `gen_space_creation_mirror_golden.py`, `make_ghost_gk_golden.py`,
> `make_xcross_directional_fixture.py`, `make_xshot_directional_fixture.py`,
> `build_lakehouse_ci_fixtures.py`, `build_synthetic_gk_fixtures.py`,
> `extract_paired_idsse_fixture.py`, `download_skillcorner_sample.py`,
> `probe_action_context_baselines.py`, `probe_tracking_baselines.py`, `profile_ac1_hotpaths.py`.
>
> Invoking those with `--help` **rewrites committed baselines, goldens and checksums, and pulls
> over the network.** An earlier draft of this task excluded only `_`-prefixed files, so all 16
> were in the parametrize and would have run **on every CI leg**, and its discovery loop ran them
> on the dev machine with no filter at all.
>
> **The population is therefore "public scripts that HAVE a parser", derived by grep — never the
> bare glob.** Note the plan's own "read `--help` first" rule could not have caught this: there is
> no parser to reject the flag.
>
> Two same-sized sets, do not conflate: **16 scripts have no argparse**; the spec's **"16 drivers
> crash on `--help`"** is a different set.

- [ ] **Step 1: Find the drivers that actually crash — with the SAME parser filter as the gate**

```bash
mkdir -p /tmp/pr5 && : > /tmp/pr5/help_fail.txt
for f in scripts/*.py; do
  b=$(basename "$f")
  case "$b" in _*) continue;; esac          # non-drivers
  grep -q "add_argument" "$f" || continue   # NO PARSER -> --help would RUN it. Never invoke.
  PYTHONIOENCODING=cp1252 python "$f" --help >/dev/null 2>/tmp/pr5/e.txt || {
    grep -q UnicodeEncodeError /tmp/pr5/e.txt && echo "$b" >> /tmp/pr5/help_fail.txt; }
done
sort -o /tmp/pr5/help_fail.txt /tmp/pr5/help_fail.txt; wc -l < /tmp/pr5/help_fail.txt
```

The `grep -q "add_argument" || continue` line is **load-bearing, not tidiness** — without it this
loop rewrites committed artifacts on the machine you run it from. Confirm the filter works before
trusting the loop:

```bash
for f in scripts/*.py; do b=$(basename "$f"); case "$b" in _*) continue;; esac; \
  grep -q "add_argument" "$f" || echo "SKIPPED (no parser): $b"; done | wc -l
```
Expected: `16`.

- [ ] **Step 2: Write the gate, and watch it fail**

Four things the earlier draft got wrong, all fixed here: assert **exit 0** (the spec's actual gate)
rather than merely the absence of one exception name; anchor the path so the parametrize list cannot
silently be **empty** (an empty parametrize collects nothing and reports success); pin the driver
population to a committed list; and **merge** the environment rather than replacing it — a stripped
env with no `SYSTEMROOT` breaks interpreter startup on the Windows CI leg.

```python
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _has_parser(p: Path) -> bool:
    """A script with NO argparse does not reject --help -- it IGNORES it and runs main().

    16 public scripts are in that class (stamp_feature_contracts, regenerate_*, gen_*, make_*,
    build_*, download_*, probe_*, profile_*). Invoking them here would rewrite committed
    baselines, goldens and SHA256SUMS, and hit the network, on EVERY CI leg. This filter is the
    only thing standing between this gate and that -- do not "simplify" it away.
    """
    return "add_argument" in p.read_text(encoding="utf-8")


DRIVERS = sorted(
    p for p in (REPO / "scripts").glob("*.py")
    if not p.name.startswith("_") and _has_parser(p)
)

assert DRIVERS, "driver list is empty -- an empty parametrize reports success vacuously"


@pytest.mark.parametrize("driver", DRIVERS, ids=lambda p: p.name)
def test_help_exits_zero_under_cp1252(driver):
    """TODO L35: --help must exit 0 on a cp1252 console."""
    proc = subprocess.run(
        [sys.executable, str(driver), "--help"],
        capture_output=True, text=True, cwd=REPO,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
```

Run: `python -m pytest tests/scripts/test_help_is_cp1252_safe.py -q > /tmp/pr5/t3.txt 2>&1; grep -E "passed|failed" /tmp/pr5/t3.txt`
Expected: **~16 FAILED**, matching `/tmp/pr5/help_fail.txt`. If the counts disagree, reconcile before
proceeding — the gate's population and the measured population must be the same set.

- [ ] **Step 3: Fix printed strings only**

argparse `help`/`description`/`epilog`, `print()`, **and the MODULE DOCSTRING of any driver that
passes `description=__doc__`** — 13 do, and that is what makes a docstring printed text. An earlier
draft of this plan said "comments and docstrings cannot crash"; that is true of comments and of
function docstrings, and **false** of a module docstring under `description=__doc__`. Both measured
crashes were exactly that.

Substitutions: `→`->`->`, `—`->`--`, `⋈`->`|><|`, `§`->`S`, `±`->`+/-`, `≥`->`>=`, `≤`->`<=`,
`×`->`x`, `σ`->`sigma`, curly quotes->straight.

**EXECUTION CORRECTION — and a retracted inference.** A first pass measured that only **2** of the
26 parser-having drivers actually crash on `--help` under cp1252, and I inferred that TODO L35's
"16" must have come from invoking the 16 parser-less scripts. **That inference was wrong.** The 16
is `_KNOWN_NON_ASCII_DRIVERS` in `tests/scripts/test_build_gkdv_arm_values.py` — a pinned debt list
with exactly 16 entries. The size match was a coincidence, and building on it was unfounded.

**A better gate already existed, so this task does not add one.** That file asserts **ASCII-only
source**, not cp1252-safety, on a measured rationale: *an em dash encodes on cp1252 and raises on
cp437; U+2265 does the opposite*. ASCII is the only set that holds under every console. The
purpose-built cp1252 test written here was weaker and redundant, and was **deleted**.

**What this task actually does: close L35 by emptying the debt list.** All 16 drivers made
ASCII-only — 62 characters, 5 distinct code points (43 `—`, 15 `§`, 2 `→`, one `Δ`, one `÷`)
substituted `--` / `S` / `->` / `delta` / `/`. The feared readability cost of de-mathing comments
did not materialise: **2 of 62** were maths symbols. `test_the_known_offender_list_is_EXACT` fails
in BOTH directions and now guards an empty list; teeth confirmed by planting a non-ASCII character
and observing `new offenders: ['probe_preprocess_baseline']`.

Two drivers additionally had the printed-text bug fixed (`description=__doc__` makes a module
docstring printed text — 13 drivers use that pattern), so their `--help` now exits 0 on cp1252.

- [ ] **Step 4: Re-run** — expected all PASS, 0 failed.

---

### Task 4: A PR-5-local scene, and gate 2 landed RED

**Files:** create `tests/tracking/test_pr5_chirality_gates.py`

**Do NOT enrich the shared `canonical_scene()`.** It is referenced across 8 of the 10
`_mirror_entries` modules (~27 references) and several entries pin **measured** tolerances to it —
`_DAS_MIRROR_TOL = 15.0` justified by a measured 12.0349, pitch-control's 7.45e-20,
`defensive_line_and_breaks`'s "defensive_line_x moves 23.75 m". Adding a defender and an attacker
changes nearest-defender distance, pitch control, DAS, team shape and line detection for every entry,
and the 10-xfail ledger this cycle is calibrated on is measured on the current scene. A two-marker
deletion would become a re-measurement of the whole 33-entry registry.

A local fixture gets N4's coverage at zero blast radius.

- [ ] **Step 1: Build `pr5_scene()` by copy-and-augment, never by mutation**

```python
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _geometry as _geo
from silly_kicks.tracking._xcross_attempt import extract_xcross_features
from silly_kicks.tracking._xshot_occurrence import extract_xshot_features
from tests.tracking._mirror_registry import canonical_scene, mirror_frames


def pr5_scene() -> pd.DataFrame:
    """One y-asymmetric frame snapshot, with a shadowing defender and an attacker in the box.

    Derived from canonical_scene()'s first frame so the 20-column schema is inherited rather than
    re-invented, then augmented on a COPY. canonical_scene() itself is never modified: 8 of the 10
    _mirror_entries modules pin measured tolerances to it (review #3, P4).
    """
    _actions, frames = canonical_scene()
    fid = frames["frame_id"].min()
    f = frames[frames["frame_id"] == fid].reset_index(drop=True).copy()

    template = f[~f["is_ball"]].iloc[0]

    def _row(pid, team, x, y, *, gk=False):
        r = template.copy()
        r["player_id"], r["team_id"] = pid, team
        r["x"], r["y"], r["is_goalkeeper"] = float(x), float(y), gk
        return r

    ball = f[f["is_ball"]].iloc[0]
    # Measured on canonical_scene() by review #3: the away/defending team id is 2 and the carrier
    # is player 11 (2.83 m from the ball). ASSERT rather than trust -- pr5_scene() must fail loudly
    # if the upstream fixture is ever renumbered, not silently score the wrong team.
    defending_team = 2
    assert (f["team_id"] == defending_team).any(), "defending team id 2 absent -- fixture changed"

    extra = pd.DataFrame([
        # shadowing defender: on the ball->goal segment, so openGoal drops below 1.0
        _row("pr5_shadow", defending_team,
             (float(ball["x"]) + 105.0) / 2.0, (float(ball["y"]) + _geo.GOAL_Y) / 2.0),
        # attacker inside the attacked penalty area, so box_off_def_ratio exceeds 0
        _row("pr5_boxatt", template["team_id"], 96.0, 30.0),
    ])
    return pd.concat([f, extra], ignore_index=True)
```

Hard-code the ids **with an assertion**, as above — so a renumbered upstream fixture fails loudly
instead of silently scoring the wrong team. Reading them off the fixture by heuristic would be worse:
a heuristic that silently picks the wrong team produces a plausible number, which is this cycle's
signature failure.

- [ ] **Step 2: Prove the fixture is non-degenerate, or record what stays degenerate**

```python
# Measured (review #3): away/defending team = 2, carrier = player 11 (2.83 m from the ball).
# Both are asserted inside pr5_scene() / the test below, so a renumbered upstream fixture fails
# loudly instead of silently scoring the wrong team or a non-existent carrier.
AWAY = 2
CARRIER = 11


def _xs(frame, goal_x):
    return extract_xshot_features(frame, gk_team_id=AWAY, goal_x=goal_x).iloc[0]


def _xc(frame, goal_x):
    return extract_xcross_features(
        frame, gk_team_id=AWAY, goal_x=goal_x,
        carrier_player_id=CARRIER, score_differential=0.0,
    ).iloc[0]


def test_pr5_scene_exercises_the_clamped_and_box_features():
    """N4: gate 2 is vacuous for any feature degenerate in BOTH legs."""
    scene = pr5_scene()
    assert (scene["player_id"] == CARRIER).any(), "carrier absent -- upstream fixture renumbered"
    assert 0.0 < _xs(scene, 105.0)["openGoal"] < 1.0
    assert _xc(scene, 105.0)["box_off_def_ratio"] > 0.0
```

Tune the two added positions until this passes. If a feature stays degenerate, record it in the
docstring as "EMPTY BY MEASUREMENT, not by omission" — never silently.

- [ ] **Step 3: Write gate 2 with the flip counts as ASSERTIONS**

The earlier draft expected "2 FAILED", which a `TypeError` also produces — so a test that never
evaluated a feature could have supplied this PR's detection-first evidence. Assert the *shape* of the
failure, not just that it failed.

```python
def _flip_report(extract_fn, goal_x_pair=(105.0, 0.0)):
    """Same physical scene, both goal ends. Post-fix every delta must be 0."""
    base_gx, mirr_gx = goal_x_pair
    base = extract_fn(pr5_scene(), base_gx)
    mirrored = extract_fn(mirror_frames(pr5_scene()), mirr_gx)
    both_finite = [
        k for k in base.index
        if np.isfinite(float(base[k])) and np.isfinite(float(mirrored[k]))
    ]
    flips = [
        k for k in both_finite
        if float(base[k]) != 0.0
        and float(base[k]) == pytest.approx(-float(mirrored[k]), rel=1e-9)
    ]
    worst = max(abs(float(base[k]) - float(mirrored[k])) for k in both_finite)
    return len(both_finite), flips, worst


@pytest.mark.parametrize(("extract_fn", "n_features"), [(_xs, 27), (_xc, 16)])
def test_features_identical_under_point_reflection(extract_fn, n_features):
    """PR 5 gate 2. The reflection is a ROTATION (x->105-x AND y->68-y), not an x mirror:
    an x-only mirror passes under the CHIRAL transform and fails under the fixed one."""
    n_compared, flips, worst = _flip_report(extract_fn)
    # NaN-shrinkage guard: if the carrier fails to resolve while tuning pr5_scene(), both_finite
    # collapses and the two assertions below pass VACUOUSLY while the test still claims to compare
    # 16 (or 27) features. n_features exists to be asserted, not to decorate the parametrize id.
    assert n_compared == n_features, f"compared {n_compared} of {n_features} -- fixture regressed"
    assert flips == [], flips
    assert worst == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 4: Run it and assert the RED has the right shape**

Run: `python -m pytest tests/tracking/test_pr5_chirality_gates.py -k point_reflection -q > /tmp/pr5/t4.txt 2>&1; grep -E "passed|failed|Error" /tmp/pr5/t4.txt`

Expected: **2 FAILED — and the failure text must list 12 flipped xS features** (`theta`, `GK_theta`,
`DefAngle_0..4`, `OffAngle_0..4`) **and 3 xCross** (`ball_theta`, `gk_theta`, `gk_lateral_offset`).

**STOP if the log contains `TypeError`, `KeyError` or `AttributeError`** — that is a broken test, not
a landed-red gate, and it is indistinguishable from success at the "2 FAILED" level. **STOP if the
counts are not 12 and 3** — the local fixture changed the defect's shape and §2's measurements no
longer describe this scene.

- [ ] **Step 5: Add the permanent non-vacuity partner (gate 3)**

```python
def test_gate2_would_fail_under_the_chiral_transform(monkeypatch):
    """Gate 3. Plant the pre-fix behaviour; gate 2 must notice. Keeps the gate proven AFTER the fix,
    where observing red once cannot."""
    monkeypatch.setattr(_geo, "to_goal_relative_y", lambda y, *, goal_x: y)
    n_compared, _flips, worst = _flip_report(_xc)
    assert n_compared == 16, f"compared {n_compared} of 16 -- the plant shrank the comparison"
    assert worst > 1e-6, "planting the chiral transform moved nothing -- gate 2 is vacuous"
```

This ERRORs until Task 6 creates `to_goal_relative_y`. That is correct ordering; it goes green in
Task 7 and must STAY green.

---

### Task 5: Gate 4 landed RED — grid centre-set symmetry

**Files:** modify `tests/tracking/test_pr5_chirality_gates.py`

The previously-specified gate ("dominant-region value equal at both ends") is **green by
construction** — after Task 8 both legs feed bit-identical goal-relative arrays. Assert the property
the anchor actually controls.

- [ ] **Step 1: Define the left-right mirror explicitly (it was referenced and never defined)**

```python
def _lr_mirror(frame: pd.DataFrame) -> pd.DataFrame:
    """y -> 68 - y at a FIXED goal end. NOT the point reflection: x is untouched, and vy negates."""
    out = frame.copy()
    out["y"] = _geo.PITCH_WIDTH - out["y"].to_numpy(dtype=float)
    if "vy" in out.columns:
        out["vy"] = -out["vy"].to_numpy(dtype=float)
    return out
```

- [ ] **Step 2: Write both halves**

```python
@pytest.mark.parametrize(("length", "axis"), [(_geo.PITCH_LENGTH, "x"), (_geo.PITCH_WIDTH, "y")])
def test_grid_centres_are_mirror_symmetric(length, axis):
    """Gate 4a. NOTE: the x half is ALREADY green today -- only y lands RED.

    Recorded so "landed red" is not read as covering both axes: ADR-051's detection-first rule is
    about observing failure, and half of this gate cannot fail.
    """
    from silly_kicks.tracking import _xcross_attempt as _xc_mod
    centres = _xc_mod._grid_centres(length, 3.0)
    assert set(np.round(centres, 9)) == set(np.round(length - centres, 9))


def test_dominant_region_is_left_right_mirror_invariant():
    """Gate 4b. Same goal end, scene mirrored left-right.

    The 5.4% / 17.74 m^2 figure in the spec was measured on canonical_scene(), NOT on pr5_scene() --
    do not assert it here and do not quote it as this fixture's number. Measure pr5_scene()'s own
    pre-fix gap in Task 5 Step 3 and record THAT as the regression signature.
    """
    a = _xc(pr5_scene(), 105.0)["space_controlled"]
    b = _xc(_lr_mirror(pr5_scene()), 105.0)["space_controlled"]
    assert float(a) == pytest.approx(float(b), abs=1e-9), f"{a} vs {b}"
```

- [ ] **Step 3: Run, confirm, and record THIS fixture's gap**

Expected: `[y]` FAILS, `[x]` PASSES, `left_right` FAILS. `_grid_centres` does not exist yet, so the
first run may ERROR on import — same red signal, resolved in Task 9.

**Record the two `space_controlled` values `left_right` reports on `pr5_scene()`.** The spec's 5.4% /
17.74 m² is `canonical_scene()`'s number and does not transfer. This fixture's own pre-fix gap is the
regression signature a future reader needs; write it into the test docstring once measured. A gap of
**zero** here means `pr5_scene()` is accidentally y-symmetric in the region the grid samples — the
gate would then be vacuous, and the fixture needs adjusting before Task 9 makes it green.

---

### Task 6: Add `to_goal_relative_y` / `to_goal_relative_vy`

**Files:** `silly_kicks/tracking/_geometry.py` (after `to_goal_relative_vx` ~`:59`; docstring `:3-6`)

- [ ] **Step 1: Add both, mirroring the existing `_x`/`_vx` shape**

```python
def to_goal_relative_y(y: float, *, goal_x: float) -> float:
    """Map absolute pitch y to goal-relative y (mirrored when the defended goal is at high x).

    Paired with :func:`to_goal_relative_x` this is the 180-degree POINT REFLECTION
    ``(x, y) -> (105 - x, 68 - y)``, so the two ends differ by a ROTATION rather than a reflection.
    Before PR 5 there was no y counterpart: ``goal_x=105`` was an x-only mirror (determinant -1) and
    ``goal_x=0`` the identity (+1), so every BEARING negated between ends while every RADIAL stayed
    byte-identical.

    Examples
    --------
    >>> to_goal_relative_y(20.0, goal_x=0.0)
    20.0
    >>> to_goal_relative_y(20.0, goal_x=105.0)
    48.0
    """
    if math.isnan(y):
        return y
    return (PITCH_WIDTH - y) if _flip(goal_x) else y


def to_goal_relative_vy(vy: float, *, goal_x: float) -> float:
    """Map absolute y-velocity to goal-relative y-velocity (negated when flipped).

    Added for symmetry with :func:`to_goal_relative_vx`. NOTE both are unused in production: no
    shipped feature consumes a directional velocity (xS ``bvx``/``bvy`` and xCross's feed only
    ``hypot``), so neither is exercised by the PR 5 feature-identity gate.

    Examples
    --------
    >>> to_goal_relative_vy(2.0, goal_x=0.0)
    2.0
    >>> to_goal_relative_vy(2.0, goal_x=105.0)
    -2.0
    """
    if math.isnan(vy):
        return vy
    return -vy if _flip(goal_x) else vy
```

Match `to_goal_relative_x`'s exact NaN-guard form (read `:41-44`) rather than assuming `math.isnan`.

- [ ] **Step 2: Correct the module docstring (`:3-6`)** — it claims LTR and RTL "map to identical
  feature values", falsified by measurement and true only after this change. Restate it as the
  invariant plus its gate (`tests/tracking/test_pr5_chirality_gates.py`), naming the pre-PR-5 counts
  (xS 12/27, xCross 3/16).

- [ ] **Step 3:** Run `python -m pytest --doctest-modules silly_kicks/tracking/_geometry.py -q`.
  Expected PASS. CI does **not** run these (private modules are ignored), so run them by hand.

---

### Task 7: xCross — convert y at the single seam

**Files:** `silly_kicks/tracking/_xcross_attempt.py:158-159`

- [ ] **Step 1: Replace the raw y read**

```python
gr_x = np.array([_geo.to_goal_relative_x(x, goal_x=goal_x) for x in f["x"].to_numpy()])
# y is GOAL-RELATIVE from here down (PR 5): paired with gr_x this is the 180-degree point
# reflection. Local name kept -- every consumer below already reads `y`.
y = np.array([_geo.to_goal_relative_y(v, goal_x=goal_x) for v in f["y"].to_numpy(dtype=float)])
```

- [ ] **Step 2: Verify** — the xCross parametrization of gate 2 PASSES; xShot still FAILS (Task 8);
  `test_gate2_would_fail_under_the_chiral_transform` PASSES.

---

### Task 8: xShot — pre-transform the frame, delete `gx`

**Files:** `silly_kicks/tracking/_xshot_occurrence.py:176-246`

`gx` is x-only and y is read at four independent sites (`:191`, `:216`, `:226`, `:236`). Patching four
sites leaves "no call site can be missed" an assertion; transforming once makes it true **by
construction** — which matters because the defect being fixed *is* a missed-site defect.

Verified safe: every x already routes through `gx()`. `DataFrame.assign` returns a new object and does
not mutate the caller's frame, as ADR-033 requires.

- [ ] **Step 1: Pre-transform once at the top of `extract_xshot_features`**

```python
# Goal-relative ONCE, for every consumer below: (x, y) -> (105 - x, 68 - y) when the defended
# goal is at high x. Do NOT reintroduce a per-site helper -- y was read at four independent
# sites, and the missed-site defect this fixes is exactly what that shape produces.
fd = frame_data.assign(
    x=frame_data["x"].map(lambda v: _geo.to_goal_relative_x(float(v), goal_x=goal_x)),
    y=frame_data["y"].map(lambda v: _geo.to_goal_relative_y(float(v), goal_x=goal_x)),
)
```

- [ ] **Step 2: Delete `gx` and repoint every reader at `fd`**

Remove `def gx(...)` (`:180-181`); take every `ball`/`defending`/`attacking`/`gk_rows` slice from
`fd`; drop the `gx` calls at `:197`, `:214`, `:224`, `:235`; rename `bx_raw` — after the
pre-transform the name is misleading.

- [ ] **Step 3: Verify no `gx` reference survives**

Run: `grep -nE "\bgx\b" silly_kicks/tracking/_xshot_occurrence.py`
Expected: **zero matches.** (`\bgx\b` cannot match `gkx` — the substring `gx` does not occur in
`g-k-x` — so "no matches other than gkx" would have been an impossible expectation.)

- [ ] **Step 4: Gate 2 goes fully green** — 0 flips, max delta 0.000e+00, both models.

- [ ] **Step 5: HARD STOP — confirm F2 with the grid still untouched**

```python
base = _xc(pr5_scene(), 105.0)["space_controlled"]
mirr = extract_xcross_features(
    mirror_frames(pr5_scene()), gk_team_id=AWAY, goal_x=0.0,
    carrier_player_id=CARRIER, score_differential=0.0,
).iloc[0]["space_controlled"]
print(base, mirr, float(base) - float(mirr))
```
Expected: identical, delta `0.0`. **If they differ, STOP** — §4's rationale is wrong, the grid is an
orientation defect after all, and the spec must be re-derived before any compute is spent.

---

### Task 9: Symmetric grid anchor + the stale cell-count docstring

**Files:** `silly_kicks/tracking/_xcross_attempt.py:104-123`

- [ ] **Step 1: Add the helper**

```python
def _grid_centres(length: float, res: float) -> np.ndarray:
    """Cell centres tiling ``length`` symmetrically about its midpoint.

    ``a = L/2 - (n-1)*res/2`` yields 1.5 for (105, 3) -- byte-identical to the shipped x grid -- and
    1.0 for (68, 3), and stays mirror-symmetric for ANY res. The shipped ``arange(res/2, L, res)`` is
    symmetric only when L divides evenly by res: true for 105/3, false for 68/3, which left the y
    centres on 34.5 instead of 34.0. Do not "simplify" back to res/2 -- at res=2.0 it is the X grid
    that becomes asymmetric, so a res-specific comment would be actively misleading.
    """
    n = int(round(length / res))
    anchor = length / 2.0 - (n - 1) * res / 2.0
    return anchor + res * np.arange(n)
```

- [ ] **Step 2: Use it for both axes**

```python
xs = _grid_centres(_geo.PITCH_LENGTH, res)
ys = _grid_centres(_geo.PITCH_WIDTH, res)
```

- [ ] **Step 3: Fix the stale docstring at `:109`** — "~1800 cells x ~22 players at res=3.0" is wrong;
  the real count is `35 x 23 = 805`. ~1800 matches `res=2.0` (52 x 34 = 1768) and went stale when
  `res` moved.

- [ ] **Step 4: Verify x is byte-identical and y moved**

```python
assert np.array_equal(_grid_centres(105.0, 3.0), np.arange(1.5, 105.0, 3.0))
assert _grid_centres(68.0, 3.0)[0] == 1.0 and len(_grid_centres(68.0, 3.0)) == 23
```

- [ ] **Step 5: Gate 4 goes green** — both parametrizations and `left_right`.

---

### Task 10: `GEOMETRY_VERSION` bump, sentinel, markers, stale prose

**Files:** `_geometry.py:25`; `tests/tracking/test_xshot_occurrence.py:381`;
`tests/tracking/_mirror_entries/trained_and_das.py` (`:11-22`, `:161`, `:218`)

A forgotten bump **fails nothing** — the mismatch path only `warnings.warn`s
(`_xshot_occurrence.py:542-548`); the fail-closed prong is on pitch dims (`:535-540`).

- [ ] **Step 1: Bump**

```python
GEOMETRY_VERSION = "goal-relative-2"  # PR 5: x-only mirror -> 180-degree point reflection
```

- [ ] **Step 2: Lift the sentinel to a module constant so it can be ASSERTED, not grepped**

In `tests/tracking/test_xshot_occurrence.py`:

```python
# Must differ from the library constant, or the test compares a value to itself.
_GEOMETRY_SENTINEL = "goal-relative-3"
```
and use it in `test_load_warns_on_geometry_version_only`.

- [ ] **Step 3: Assert both behaviourally**

The earlier draft grepped the test file's source for `"goal-relative-3"`. CLAUDE.md is explicit that
keyword/substring tests over source are not evidence of behaviour — it would pass on a comment. A gate
that exists to prevent green-by-construction decay must not itself be green by construction.

```python
def test_geometry_version_was_bumped_for_the_point_reflection():
    """Gate 5. A forgotten bump fails nothing -- the mismatch path only warns."""
    assert _geo.GEOMETRY_VERSION == "goal-relative-2"


def test_geometry_sentinel_still_differs_from_the_library_constant():
    from tests.tracking.test_xshot_occurrence import _GEOMETRY_SENTINEL
    assert _GEOMETRY_SENTINEL != _geo.GEOMETRY_VERSION
```

- [ ] **Step 4: Delete the two strict markers AND the prose denying them**

Remove `defect=(...)` at `trained_and_das.py:161` and `:218`. Delete `:11-22`, which claims the
entries are "Registered at the exact tolerance **with no defect marker**" and "NOTE (finding,
deliberately NOT xfail-ed)" — contradicted by the `defect=` args. The module note describing the
transform as x-only also goes stale on merge.

- [ ] **Step 5: Verify the ledger dropped 10 -> 8, no XPASS**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -q -rxX --no-header > /tmp/pr5/t10.txt 2>&1; grep -cE "^XFAIL" /tmp/pr5/t10.txt; grep -cE "^XPASS" /tmp/pr5/t10.txt`
Expected: `8` and `0`. The survivors are all Gate B / D3 (PR 6). An XPASS fails the build.

---

### Task 11: Gates, `/final-review`, COMMIT 1

- [ ] **Step 1: Lint, format, types**

```bash
python -m ruff check . > /tmp/pr5/ruff.txt 2>&1; echo "ruff=$?"
python -m ruff format --check . > /tmp/pr5/fmt.txt 2>&1; echo "fmt=$?"
python -m pyright > /tmp/pr5/pyright.txt 2>&1; echo "pyright=$?"
```
Expected all `=0`. Run `pyright` **bare** — CI gates `tests/` too.

- [ ] **Step 2: Full non-e2e suite — "0 failed" is IMPOSSIBLE here, by design**

Expected: **8 xfailed, no XPASS**, and a bounded set of failures **all** traceable to the bundled
weights predating the transform. That is §5.1's atomicity observed, not a regression: the chirality
guard recomputes the model's output on the canonical probe and raises when it moves — measured
`0.0054975743` (stored) -> `0.0080407225` (recomputed).

The gate is therefore **not** "zero failures" but **"every failure is a fail-closed weights guard,
and the count matches the enumerated list"**. Record the list before commit 1 and check it back to
zero after commit 2. A failure that is NOT a chirality/feature-contract raise is a real regression
and must be fixed before committing.

- [ ] **Step 3: C4** — PASS, no aggregator count change (PR 5 adds no `add_*`).

- [ ] **Step 4: `/final-review`**, fix findings in the working tree.

- [ ] **Step 5: Present the diff, get explicit approval, COMMIT 1**

Stage by explicit path — never `git add -A`. **Include the spec and this plan**; they are part of
commit 1, and leaving them untracked would make every fit in Phase B refuse.

- [ ] **Step 6: Verify the tree is genuinely clean**

```bash
git status --porcelain
```
Expected: **empty output.** No `grep -v` filter — a pre-check that filters what the real gate refuses
on is worse than no check.

Commit 1 is legitimately broken on its own: its bundled weights refuse to load, by design (§5.1).
That is what makes it a citable clean SHA.

---

## PHASE B — fits and stamps (Commit 2)

### Task 12: Four fits + the platform `atol` measurement

- [ ] **Step 1: Confirm clean, and capture the SHA the artifacts will cite**

```bash
git status --porcelain          # MUST be empty
git rev-parse HEAD | tee /tmp/pr5/commit1.sha
```

- [ ] **Step 2: Confirm the `sc_extended` corpus before spending compute**

Run: `grep -n "sc_extended" scripts/_paired.py | head -20`. Record the real match count — spec §5.3
flags it as inferred.

- [ ] **Step 3: Read each trainer's `--help`** — the arm selector's flag name is **not assumed**.

- [ ] **Step 4: Run the four fits, outputs OUTSIDE the repo**

```bash
export PINING_CACHE="$HOME/.cache/silly-kicks-pining"; mkdir -p "$PINING_CACHE"
export PR5_RUNS="$HOME/runs/pr5"; mkdir -p "$PR5_RUNS"
for m in xshot_occurrence xcross_attempt; do
  for arm in public sc_extended; do
    python scripts/train_${m}.py --variant "$arm" --cache-dir "$PINING_CACHE" \
      --output-dir "$PR5_RUNS/${m}_${arm}" > "/tmp/pr5/fit_${m}_${arm}.log" 2>&1
    echo "${m} ${arm} exit=$?"
  done
done
```

`--output-dir` **must** be outside the repo: inside it creates `?? runs/` and the next
`require_clean_tree` refuses, breaking the loop at iteration 2.

Confirm each `metrics.json` has `run_commit` == `/tmp/pr5/commit1.sha` and `run_tree_dirty: false`.

- [ ] **Step 5: Measure the DGX-vs-x86 delta (for PR 7)**

Run on **both** platforms and diff:

```python
import json
from silly_kicks.tracking._feature_contract import contract_probe_frame
from silly_kicks.tracking._xshot_occurrence import extract_xshot_features

vec = extract_xshot_features(contract_probe_frame(), gk_team_id="B", goal_x=105.0).iloc[0]
print(json.dumps({k: float(v) for k, v in vec.items()}, sort_keys=True))
```

`gk_team_id="B"` is not a guess — it is what `_xshot_occurrence.py:355`'s `_vec()` fingerprints, so
any other value produces a vector that is not comparable to the stored one and the "platform delta"
would really be a probe-argument delta.

**`gk_team_id` must match the frame the stored contract was built from**, or the vectors are not
comparable and the "platform delta" is really a probe delta. Do not hard-code it on trust — recompute
and check against the artifact:

```python
import json
import numpy as np
from pathlib import Path

meta = json.loads(Path("silly_kicks/tracking/_xshot_weights/default/metadata.json").read_text("utf-8"))
stored = np.asarray(meta["feature_contract"]["fingerprint"], dtype=float)   # the rounded 27-vector

# ELEMENTWISE, not a length check: 27 == 27 holds for ANY gk_team_id, so a length assertion
# cannot discriminate the thing this is here to catch. atol/rtol are the contract's OWN
# (_feature_contract.py:41-42), so the check is calibrated to what it guards.
assert np.allclose(vec.to_numpy(dtype=float), stored, atol=1e-6, rtol=0.0), (
    "x86 probe does not reproduce the stored fingerprint -- this is a probe-argument error, "
    "NOT a platform delta. Fix it before treating any DGX difference as meaningful."
)
```

Run this on **x86 first**. It must reproduce exactly there, because that is the stamping platform;
only once it does is a DGX difference attributable to the platform.

**Write the two JSON files OUTSIDE the repo** — `$PR5_RUNS/platform_atol/{dgx,x86}.json`. Writing
into `docs/research/` here would dirty the tree in the middle of Phase B, so Task 13 Step 6's
"porcelain must be empty" could not pass and Task 14's probe would refuse — the same failure the
three-commit split exists to prevent, one layer down. The files are copied into
`docs/research/pr5_platform_atol/` in **Task 15**, as part of commit 3.

Per ADR-050 §1 the answer may not be "widen the number": if a covering tolerance would also swallow a
real 1 cm geometry change, the honest conclusion is that fingerprints are **platform-scoped**. Record
which the measurement supports.

- [ ] **Step 6: Verify `artifact_label`** — `public` for the two public arms, restricted-tier for the
  two `sc_extended` arms, produced by ADR-038 rather than assigned by hand (§5.3).

---

### Task 13: Re-stamp on x86, verify fail-closed loading, COMMIT 2

- [ ] **Step 1: Copy weights to x86 and stamp there** — keeps every fingerprint x86-produced, the
  policy `_feature_contract.py:37-40` already records. Confirm the runbook does the copy (§5.4).

**`stamp_feature_contracts.py` takes NO arguments** — verified, zero `argparse` occurrences; `main()`
iterates a module-level `TARGETS` unconditionally. Any flag you pass is silently ignored, the run
looks like it succeeded, and it stamps **all three** artifacts including `_ghost_gk_weights/default`,
which §5.2 places outside this PR's blast radius. The plan's "read `--help` first" rule cannot catch
this class — there is no parser to reject the flag.

Run: `python scripts/stamp_feature_contracts.py`

Then **verify the blast radius held**, rather than assuming ghost was a no-op:

```bash
git diff --name-only silly_kicks/tracking/_ghost_gk_weights/ \
                     silly_kicks/tracking/_xshot_weights/ \
                     silly_kicks/tracking/_xcross_weights/
```
Expected: only paths under `_xshot_weights/` and `_xcross_weights/`. **If any
`_ghost_gk_weights/` path appears, STOP** — ghost does not consume the chiral transform, so a
changed ghost stamp means either the stamper is non-deterministic or something outside this PR's
scope moved. Restore it with `git checkout -- silly_kicks/tracking/_ghost_gk_weights/` and find out
why before continuing.

- [ ] **Step 2: Verify the stamps**

```python
import json
from pathlib import Path

for weights in ("_xshot_weights", "_xcross_weights"):
    p = Path("silly_kicks/tracking") / weights / "default" / "metadata.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    assert m["geometry_version"] == "goal-relative-2", (p, m["geometry_version"])
    assert "chirality" in m and "feature_contract" in m, p
    print(p, "OK")
```

`SHA256SUMS` changes alongside `metadata.json` — a checksum-pinning consumer sees a diff.

- [ ] **Step 3: Prove the artifacts load**

Run: `python -m pytest tests/ -k "weights_bundle_golden or chirality or feature_contract" -q > /tmp/pr5/t13.txt 2>&1; grep -E "passed|failed" /tmp/pr5/t13.txt`
Expected: PASS. This is where code, weights and stamps are proven to agree — if any drifted, one of
the two fail-closed guards raises here.

- [ ] **Step 4: Upload the `sc_extended` arms**

Targets `silly-kicks/xshot-occurrence-v1` and `silly-kicks/xcross-attempt-v1`, served at the repo
ROOT to match `from_hub`. Then verify `from_variant("sc_extended")` round-trips for both — before
this PR they raise, their chirality stamps having been computed on the old transform. Update both
model cards with the corrected geometry and the `artifact_label`.

- [ ] **Step 5: Full gates, `/final-review`, present the diff, COMMIT 2**

- [ ] **Step 6: `git status --porcelain` must be empty** before Phase C.

---

## PHASE C — probes and docs (Commit 3)

### Task 14: Re-run the two registered TF-19 probes

This is why commit 2 exists: `validate_xs_probe.py:438` calls `require_clean_tree`, and the probe must
run **with** the new weights. Now it does, and each research directory cites a SHA that actually
contains them.

- [ ] **Step 1: Read `--help` first** — `--out` is `required=True` (`:390`); omitting it exits before
  doing anything. `--variant` and `--lock-commit` exist; `--lock-commit` defaults to HEAD.

```bash
python scripts/validate_xs_probe.py --variant both --lock-commit 78ffc70 \
  --out docs/research/tf19_pr3b_xs_v2 > /tmp/pr5/xs_probe.log 2>&1
```

Confirm `78ffc70` against `docs/research/tf19_pr3b_xs_v2/metrics.json` before running — the point of
citing a lock commit is that it is not retyped from memory.

Then the xCross substitution probe (`gk_substitution_probe`, `_xcross_eval.py`). **Do not re-tune** —
the constants are pre-registered; re-running is legitimate, re-tuning is not.

- [ ] **Step 2: Update the six directories** — `tf19_entanglement`, `tf19_pr2`, `tf19_pr3b`,
  `tf19_pr3b_xs_v2`, `tf19_signoff_power`, `xcross_causal` — each with the new `run_commit` and
  figures.

- [ ] **Step 3: Record the verdict whichever way it goes.** **A verdict may flip. That is a result,
  not a failure of the PR.** If `tf19_ready` or the xS-v2 verdict changes, record it plainly and
  surface it — do not quietly keep the old text.

---

### Task 15: Docs, commit-prep, COMMIT 3, PR

- [ ] **Step 1: Bring the platform-`atol` measurement into the repo**

It was written to `$PR5_RUNS/platform_atol/` in Task 12 Step 5 precisely so it would not dirty the
tree mid-Phase-B. It belongs to commit 3:

```bash
mkdir -p docs/research/pr5_platform_atol
cp "$PR5_RUNS"/platform_atol/{dgx,x86}.json docs/research/pr5_platform_atol/
```

Write that directory's `README.md` with the max abs delta, which platform stamped, and — per ADR-050
§1 — whether the measurement supports widening `atol` or declaring fingerprints **platform-scoped**.
PR 7 consumes this by path.

- [ ] **Step 2: CHANGELOG + CLAUDE.md retrain-trigger declaration — the SHAPE, not just the fact**

> `xshot_occurrence_xfns` / `xcross_attempt_xfns` are wired into `pre_shot_gk_full_default_xfns`, so
> opted-in VAEP consumers re-materialize. The transform fix moves **only rows attacking the high-x
> goal** — `to_goal_relative_y(y, goal_x=0)` is the identity — roughly half the corpus, precisely the
> home-vs-away split inside a single match. The grid change is **two-sided**: on the fixture 0% at one
> end and −5.4% at the other, scene-dependent rather than a structural rule.

CLAUDE.md is at 79k against a 150k limit, but the C4 box-description cap (200 chars) has failed CI
before.

- [ ] **Step 3: Full gates** (Task 11 Steps 1–3)

- [ ] **Step 4: `/final-review`**, fix findings in the working tree

- [ ] **Step 5: Commit-prep — merge `origin/main` FIRST, then number**

```bash
git fetch origin && git merge origin/main
```
Only then assign version / PR-S / ADR from the merged state. Five collisions in this cycle: checking
just before writing the number bounds the window but does not close it. Write the version to **all
five** sites: `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock` (via `uv lock`), CHANGELOG
heading, TODO "Current release".

- [ ] **Step 6: Present the diff, get explicit approval, COMMIT 3**

- [ ] **Step 7: PR from the feature branch — request `--merge`, NOT squash**

The repo default is squash-only. A squash rewrites commit 1's and commit 2's SHAs and orphans the
`run_commit` citations in four wheel artifacts and six research directories. Same reason 4.73.0 was
merged non-squash.

---

## Open items carried into execution

1. **Nothing is blocked.** Task 2's licensing question was ruled on 2026-08-02 — wire the guard in
   and ship the `"full"` label; see the Task 2 header for the reasoning. (An earlier revision of this
   list said Task 2 was blocked, which contradicted the task itself. One decision made two statements
   false and only one was updated — the same stale-together pattern the repo records for registered
   constants.)
2. **`sc_extended` match count** is inferred; confirm at Task 12 Step 2 before spending compute.
3. **Task 4 Step 4 and Task 8 Step 5 are hard stops.** A `TypeError` in the gate is not a landed-red
   gate; a non-zero `space_controlled` delta means §4's rationale is wrong.
4. **`pr5_scene()`'s added positions need tuning** until Task 4 Step 2 passes. If a feature stays
   degenerate, record it — never leave it silently untested inside a gate.
