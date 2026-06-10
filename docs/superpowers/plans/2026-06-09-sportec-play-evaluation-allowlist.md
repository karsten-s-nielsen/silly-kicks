# DFL `play_evaluation` success-allowlist (sportec) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sportec exact-failure-match (`play_evaluation == "unsuccessful"`) with a guarded, kloppy-aligned **success-allowlist**, single-sourced across both completion sites, so any unseen reason-coded DFL failure token is handled by construction.

**Architecture:** Three module-level helpers in `silly_kicks/spadl/sportec.py` — `_extract_play_eval` (DataFrame→clean-str-array adapter), pure `_play_evaluation_is_fail` (success-allowlist classifier), side-effecting `_warn_unexpected_play_eval` (observability) — that both the main pass/set-piece site and the synth-distribution site route through. Behavior is byte-identical on observed DFL data (the only non-success token is `unsuccessful`); the change hardens against unseen failure tokens and aligns the native converter with the kloppy gateway. A CI-everywhere native-shape regression test plus an owner-gated Databricks-bronze e2e on the 7 IDSSE matches form the regression net.

**Tech Stack:** Python, numpy, pandas, pytest; `scripts._loader_databricks` (PEP-420 namespace pkg) + `databricks-sql-connector` (lazy, owner-only) for the e2e.

**Spec:** `docs/superpowers/specs/2026-06-09-sportec-play-evaluation-allowlist-design.md` (v3).

---

## Commit policy (read first)

This branch (`pr-s90-sportec-play-evaluation-allowlist`, already created) produces **one commit at the
end** (Task 7), gated on **explicit user approval + the git-commit sentinel** (standing policy + the
sentinel hook). Tasks 0–6 stage changes and run tests/lint but **do NOT commit**.

## File Structure

- **Modify** `silly_kicks/spadl/sportec.py` — add the three module-level helpers (before
  `_build_raw_actions`, ~line 737); rewire Site 1 (`~855-858`) and Site 2 (`~1069-1077`).
- **Modify** `tests/spadl/test_sportec_completion.py` — add the genuinely-new behavioral cases + the
  single-source agreement test + the committed native-shape distribution/warn regression.
- **Modify** `scripts/_loader_databricks.py` — add the `fetch_idsse_events()` bronze read helper.
- **Create** `tests/spadl/test_sportec_playeval_e2e.py` — owner-gated Databricks-bronze e2e.
- **Modify** `TODO.md`, version sites (`pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `uv.lock`).

---

## Task 0: Orientation + green baseline

**Files:** none modified.

- [ ] **Step 1: Sync to latest main, confirm the branch, capture the baseline**

Run: `git fetch origin main && git branch --show-current` → expect `pr-s90-sportec-play-evaluation-allowlist`.
Confirm the branch contains the latest `origin/main` (so the loader anchor, the 4.21.2→4.21.3 bump, and the
`[4.21.2]` CHANGELOG insertion still hold):
`git merge-base --is-ancestor origin/main HEAD && echo "up to date" || echo "REBASE NEEDED"`.
If `REBASE NEEDED`, rebase onto `origin/main` and re-derive the version/CHANGELOG anchors (the next free
patch may have shifted past 4.21.3) before proceeding.
Run: `python -m pytest tests/spadl/test_sportec_completion.py -q 2>&1 | tail -5`
Expected: all pass (record the count). This is the pre-change baseline for the sportec completion file.

---

## Task 1: The three helpers + rewire both sites (red → green)

**Files:**
- Modify: `silly_kicks/spadl/sportec.py`
- Test: `tests/spadl/test_sportec_completion.py`

- [ ] **Step 1: Write the failing tests (red-first for all three new behaviors)**

Append to `tests/spadl/test_sportec_completion.py` — one test per new behavior (main classification,
the warn, the synth site), so each is genuinely test-driven, not green-on-arrival:

```python
def test_unseen_reason_coded_failure_is_fail():
    # Main path: any non-empty, non-success token (e.g. a reason-coded failure the 4.20.1 exact-match
    # left as success) -> fail. The headline new behavior.
    actions = _convert(_ev([dict(event_type="Play", play_evaluation="unsuccessfulBecauseOfFoul")]))
    assert actions[actions["type_id"] == _PASS].iloc[0]["result_id"] == _FAIL


def test_unexpected_token_warns_and_fails():
    # The warn (observability): an unexpected non-success token is surfaced, not silently classified.
    with pytest.warns(UserWarning, match="unexpected play_evaluation"):
        actions = _convert(_ev([dict(event_type="Play", play_evaluation="weirdNovelToken")]))
    assert actions[actions["type_id"] == _PASS].iloc[0]["result_id"] == _FAIL


def test_synth_goalkick_unseen_token_is_fail():
    # Synth site: the punt-synthesized goalkick inherits the parent Play's eval via the SAME allowlist.
    actions = _convert(
        _ev([dict(event_type="Play", play_goal_keeper_action="punt", play_evaluation="unsuccessfulBecauseOfFoul")])
    )
    assert actions[actions["type_id"] == _GOALKICK].iloc[0]["result_id"] == _FAIL
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/spadl/test_sportec_completion.py -q -k "unseen or unexpected_token"`
Expected: all three FAIL — current code maps unseen tokens to `_SUCCESS` (exact `== "unsuccessful"`)
and emits no warning (no warn helper exists yet).

- [ ] **Step 3: Add the three helpers**

In `silly_kicks/spadl/sportec.py`, insert immediately **before** `def _build_raw_actions(` (~line 737):

```python
_SUCCESS_EVAL: tuple[str, str] = ("successfullyCompleted", "successful")
# Everything we expect on a DFL pass/set-piece Evaluation; anything else is surfaced by the warn.
_KNOWN_EVAL: frozenset[str] = frozenset(_SUCCESS_EVAL) | {"unsuccessful", ""}


def _extract_play_eval(df: pd.DataFrame) -> np.ndarray:
    """Normalize the optional DFL ``play_evaluation`` column to a clean str array.

    Absent column / NaN / null -> ``""`` so a missing column never mass-fails non-DFL sportec-like
    data. Single source of the extraction both completion sites share (no per-site ``fillna`` drift).
    """
    if "play_evaluation" in df.columns:
        return df["play_evaluation"].fillna("").astype(str).to_numpy()
    return np.full(len(df), "", dtype=object)


def _play_evaluation_is_fail(play_eval: np.ndarray) -> np.ndarray:
    """Success-allowlist completion: non-empty, non-success DFL Evaluation -> fail (kloppy-aligned).

    Empty / absent / null -> not-fail (the conservative success default). Any unseen reason-coded
    failure token (e.g. ``unsuccessfulBecauseOfFoul``) -> fail by construction. Exact camelCase match
    (DFL is consistent camelCase; a case-variant fails+warns by design -- deliberately NOT
    ``.str.lower()`` like sibling qualifiers). Mirrors kloppy ``sportec/deserializer.py`` (the
    reference DFL parser): ``Evaluation in {successfullyCompleted, successful}`` is the success set.
    """
    return (play_eval != "") & ~np.isin(play_eval, _SUCCESS_EVAL)


def _warn_unexpected_play_eval(play_eval: np.ndarray) -> None:
    """Surface any token that is neither a known success nor the known ``unsuccessful`` failure.

    Makes a genuinely-new or benign DFL token visible (so it can be added to ``_SUCCESS_EVAL``)
    instead of silently classified as fail. Called at BOTH completion sites over each site's relevant
    rows -- punt-Play synth parents are excluded from ``is_pass``, so one warn cannot cover both.
    """
    unexpected = set(np.unique(play_eval)) - _KNOWN_EVAL
    if unexpected:
        warnings.warn(
            f"sportec: unexpected play_evaluation token(s) {sorted(unexpected)} treated as fail "
            f"(not in the success allowlist {_SUCCESS_EVAL}); verify against the DFL spec.",
            stacklevel=2,
        )
```

- [ ] **Step 4: Rewire Site 1 (main pass + set-piece) — comment + code together**

In `_build_raw_actions`, replace the **whole block including the now-false BUG-2 comment**
(lines ~849-858 — the comment currently asserts "The lone failure token is `unsuccessful` … only an
explicit `unsuccessful` flips to fail", which the allowlist makes false; leaving it is a doc-drift /
Chesterton's-Fence trap). Leave the `is_goalkick` definition at ~845 untouched. Replace:

```python
    # BUG-2 fix (2026-06-09): pass-class + set-piece completion comes from the native DFL
    # `play_evaluation` (carried on Play AND on GoalKick/FreeKick/Corner/ThrowIn via their nested
    # Play; confirmed on 7 real DFL matches). Previously all of these were hard-wired success,
    # zeroing failed-pass / failed-goalkick labels (IDSSE goalkicks read 100% success vs the real
    # ~71%). The lone failure token is `unsuccessful`; `successfullyCompleted`/`successful` and
    # NULL/unknown stay success (conservative -- only an explicit `unsuccessful` flips to fail).
    play_eval = _opt("play_evaluation", "").fillna("").astype(str).to_numpy()
    is_eval_fail = play_eval == "unsuccessful"
    is_pass_or_setpiece = is_pass | is_freekick | is_corner | is_throwin | is_goalkick
    result_ids[is_pass_or_setpiece & is_eval_fail] = spadlconfig.result_id["fail"]
```

with:

```python
    # Pass-class + set-piece completion from the native DFL `play_evaluation` (carried on Play AND on
    # GoalKick/FreeKick/Corner/ThrowIn via their nested Play). Success-ALLOWLIST (kloppy-aligned, 4.21.3):
    # `fail` iff the Evaluation is non-empty AND not in {successfullyCompleted, successful} -- so any
    # unseen reason-coded failure token (e.g. `unsuccessfulBecauseOfFoul`) fails by construction, while
    # empty/absent/NULL stays success (conservative; never mass-fails a missing column). Byte-identical
    # on observed DFL data (the only non-success token across the 7 IDSSE matches is `unsuccessful`).
    play_eval = _extract_play_eval(rows)
    is_pass_or_setpiece = is_pass | is_freekick | is_corner | is_throwin | is_goalkick
    result_ids[is_pass_or_setpiece & _play_evaluation_is_fail(play_eval)] = spadlconfig.result_id["fail"]
    _warn_unexpected_play_eval(play_eval[is_pass_or_setpiece])
```

- [ ] **Step 5: Rewire Site 2 (synth distribution) — comment + code together**

In `_synthesize_gk_distribution_actions`, replace the **comment + code block** (lines ~1066-1077; the
comment's "only an explicit `unsuccessful` is a fail" is now false). Replace:

```python
    # BUG-2 fix (2026-06-09): the synthesized distribution (throwOut->pass / punt->goalkick)
    # inherits the parent Play's native completion (play_evaluation); only an explicit
    # `unsuccessful` is a fail (mirrors the open-play / set-piece rule above).
    if "play_evaluation" in src.columns:
        synth_eval = src["play_evaluation"].fillna("").astype(str).to_numpy()
    else:
        synth_eval = np.full(n_synth, "", dtype=object)
    result_ids_synth = np.where(
        synth_eval == "unsuccessful",
        spadlconfig.result_id["fail"],
        spadlconfig.result_id["success"],
    ).astype(np.int64)
```

with:

```python
    # The synthesized distribution (throwOut->pass / punt->goalkick) inherits the parent Play's
    # native completion via the SAME success-allowlist as the main path (single-sourced helpers);
    # empty/absent -> success. Mirrors the open-play / set-piece rule above.
    synth_eval = _extract_play_eval(src)
    result_ids_synth = np.where(
        _play_evaluation_is_fail(synth_eval),
        spadlconfig.result_id["fail"],
        spadlconfig.result_id["success"],
    ).astype(np.int64)
    _warn_unexpected_play_eval(synth_eval)
```

- [ ] **Step 6: Run the three new tests + the existing suite to verify green**

Run: `python -m pytest tests/spadl/test_sportec_completion.py -q`
Expected: all pass, including the three new red-first tests (main unseen→fail, unexpected→warns,
synth unseen→fail) and the pre-existing `unsuccessful`/`successfullyCompleted`/`successful`/`""`
cases (byte-identical on those).

- [ ] **Step 7: Lint the changed file**

Run: `python -m ruff check silly_kicks/spadl/sportec.py && python -m ruff format --check silly_kicks/spadl/sportec.py`
Expected: no errors.

---

## Task 2: Remaining behavioral cases + single-source guard

**Files:**
- Test: `tests/spadl/test_sportec_completion.py`

- [ ] **Step 1: Add the cases**

Append to `tests/spadl/test_sportec_completion.py`:

```python
def test_play_evaluation_column_absent_no_mass_fail():
    # No play_evaluation key at all (column absent) must NOT mass-fail passes (the allowlist trap).
    actions = _convert(_ev([dict(event_type="Play")]))
    assert actions[actions["type_id"] == _PASS].iloc[0]["result_id"] == _SUCCESS


def test_known_tokens_are_warn_silent():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning -> test failure
        for tok in ("successfullyCompleted", "successful", "unsuccessful", ""):
            _convert(_ev([dict(event_type="Play", play_evaluation=tok)]))


@pytest.mark.parametrize(
    "evaluation", ["successfullyCompleted", "successful", "unsuccessful", "unsuccessfulBecauseOfFoul", ""]
)
def test_main_and_synth_paths_agree(evaluation):
    # Single-source guard: the main pass path and the synth-goalkick path must map every token to the
    # same result_id (both route through _play_evaluation_is_fail). A drift would break one side.
    main = _convert(_ev([dict(event_type="Play", play_evaluation=evaluation)]))
    main_res = main[main["type_id"] == _PASS].iloc[0]["result_id"]
    synth = _convert(
        _ev([dict(event_type="Play", play_goal_keeper_action="punt", play_evaluation=evaluation)])
    )
    synth_res = synth[synth["type_id"] == _GOALKICK].iloc[0]["result_id"]
    assert main_res == synth_res
```

- [ ] **Step 2: Run to verify they pass**

Run: `python -m pytest tests/spadl/test_sportec_completion.py -q`
Expected: all pass. (These exercise paths the Task 1 implementation already covers — they are
regression coverage; none should fail. If `test_main_and_synth_paths_agree` fails, the two sites have
drifted — re-check Step 4/5 of Task 1.)

---

## Task 3: Committed native-shape distribution + warn regression (M3)

**Files:**
- Test: `tests/spadl/test_sportec_completion.py`

> The native converter takes a pre-parsed DataFrame, not XML (no native DFL-XML parser exists — that
> is the deferred TF-23), so this committed regression uses the `_ev(...)` native-shape builder with
> the full observed DFL vocabulary, NOT the kloppy-gateway `tests/datasets/kloppy/sportec_events.xml`.

- [ ] **Step 1: Add the distribution/warn regression test**

Append to `tests/spadl/test_sportec_completion.py`:

```python
def test_observed_distribution_regression_and_single_batch_warn():
    # Lock "robustness hardening, not re-mapping": the full observed DFL vocabulary + one reason-coded
    # token, in one converter pass. Clean tokens map byte-identically to the 4.20.1 exact-match
    # converter; the reason-code -> fail; exactly the one unexpected token is named in a single warn.
    rows = [
        dict(event_type="Play", play_evaluation="successfullyCompleted"),
        dict(event_type="Play", play_evaluation="successful"),
        dict(event_type="Play", play_evaluation="unsuccessful"),
        dict(event_type="Play", play_evaluation=""),
        dict(event_type="Play", play_evaluation="unsuccessfulBecauseOfFoul"),
    ]
    with pytest.warns(UserWarning, match=r"unsuccessfulBecauseOfFoul"):
        actions = _convert(_ev(rows))
    passes = actions[actions["type_id"] == _PASS].reset_index(drop=True)
    assert list(passes["result_id"]) == [_SUCCESS, _SUCCESS, _FAIL, _SUCCESS, _FAIL]
```

- [ ] **Step 2: Run to verify it passes**

Run: `python -m pytest tests/spadl/test_sportec_completion.py::test_observed_distribution_regression_and_single_batch_warn -q`
Expected: PASS.

- [ ] **Step 3: Lint the test file**

Run: `python -m ruff check tests/spadl/test_sportec_completion.py && python -m ruff format --check tests/spadl/test_sportec_completion.py`
Expected: no errors.

---

## Task 4: Bronze read helper + owner-gated e2e

**Files:**
- Modify: `scripts/_loader_databricks.py`
- Create: `tests/spadl/test_sportec_playeval_e2e.py`

- [ ] **Step 1: Add the `fetch_idsse_events` read helper**

In `scripts/_loader_databricks.py`, after `fetch_action_values` / `shape_action_values` (before
`_convert`, ~line 144), add:

```python
def fetch_idsse_events() -> pd.DataFrame:
    """Read all bronze IDSSE event rows (native sportec-converter input shape) for the owner-gated
    play_evaluation e2e. Read-only; the 7 public IDSSE matches are ~10.5k events. Table name comes
    from the allowlist-validated ``_table`` (idsse is in ``_ALLOWED_PROVIDERS``) -- no user input.
    """
    conn = _connect()
    try:
        cur = conn.cursor()
        return _query_param(cur, f"SELECT * FROM {_table('idsse', 'events')}")  # noqa: S608
    finally:
        conn.close()
```

- [ ] **Step 2: Lint the loader**

Run: `python -m ruff check scripts/_loader_databricks.py && python -m ruff format --check scripts/_loader_databricks.py`
Expected: no errors. (No unit test — pure I/O over the connector, like `fetch_action_values`.)

- [ ] **Step 3: Create the owner-gated e2e**

Create `tests/spadl/test_sportec_playeval_e2e.py`:

```python
"""Owner-gated e2e: the native sportec play_evaluation success-allowlist on the real 7 IDSSE matches.

Runs the NATIVE silly_kicks.spadl.sportec converter on Databricks bronze.idsse_events (the only path
that surfaces the raw play_evaluation token to the converter this PR changes -- pining's IDSSE loader
parses via the kloppy gateway, which never exposes it). Asserts the success-allowlist is warn-silent
on real DFL data (allowlist u {unsuccessful} covers the vocabulary -> byte-identical to 4.20.1) and
the BUG-2 mechanism is still live (goalkick fail-rate in a plausible band). Skips in public CI; needs
the owner Databricks credentials + databricks-sql-connector (install in an isolated env, NOT the main
.venv -- the connector pins pandas<2.3.0). See docs/superpowers/specs/2026-06-09-sportec-play-evaluation-allowlist-design.md.
"""

import importlib.util
import os
import warnings

import pytest

import scripts._loader_databricks as L
import silly_kicks.spadl.config as spadlconfig
from silly_kicks.spadl import sportec as sportec_spadl

# NOTE: deliberately NO `silly_kicks.tracking` import -- it transitively pulls xgboost + numba +
# sklearn (verified), which a SPADL-completion e2e must not drag in. ET is dropped inline below.

_DBX_ENV = ("DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN")
_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_FAIL = spadlconfig.result_id["fail"]


def _connector_available() -> bool:
    try:
        return importlib.util.find_spec("databricks.sql") is not None
    except ModuleNotFoundError:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not all(os.environ.get(k) for k in _DBX_ENV),
        reason="owner-tier Databricks credentials (DATABRICKS_HOST/HTTP_PATH/TOKEN)",
    ),
    pytest.mark.skipif(
        not _connector_available(),
        reason="databricks-sql-connector not importable (install in an isolated env, NOT the main .venv)",
    ),
]


def test_play_evaluation_allowlist_on_real_idsse(capsys):
    raw = L.fetch_idsse_events()
    assert len(raw) > 0, "bronze.idsse_events returned no rows"
    assert "play_evaluation" in raw.columns, "bronze.idsse_events missing play_evaluation"

    gk_total = 0
    gk_fail = 0
    caught: list[str] = []
    for match_id, ev in raw.groupby("match_id"):
        # Defensive ET drop (Bundesliga has no ET, but the native converter RAISES on ET-without-flag,
        # ADR-010). Inline + dtype-robust (drops periods 3/4 whether int or str) -> no tracking import.
        ev = ev[~ev["period"].astype(str).isin(["3", "4"])]
        if ev.empty:
            continue
        # home_team_id is orientation-only; result_id from play_evaluation is orientation-independent.
        home = str(ev["team"].dropna().mode().iloc[0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            actions, _ = sportec_spadl.convert_to_actions(ev, home_team_id=home, home_team_start_left=True)
        caught += [str(x.message) for x in w if "unexpected play_evaluation" in str(x.message)]
        gk = actions[actions["type_id"] == _GOALKICK]
        gk_total += len(gk)
        gk_fail += int((gk["result_id"] == _FAIL).sum())

    fail_rate = gk_fail / gk_total if gk_total else float("nan")
    with capsys.disabled():
        print("\n=== sportec play_evaluation allowlist e2e (bronze IDSSE) ===")
        print(f"matches={raw['match_id'].nunique()}  goalkicks={gk_total}  goalkick_fail_rate={fail_rate:.3f}")
        print(f"observed raw play_evaluation values: {sorted(set(raw['play_evaluation'].dropna().astype(str)))}")

    # (1) Warn-silent: allowlist u {unsuccessful} covers the real vocabulary (byte-identical condition).
    assert not caught, f"unexpected play_evaluation token(s) on real IDSSE: {caught}"
    # (2) Liveness band: BUG-2 mechanism live (fails exist) AND not an all-fail regression.
    assert gk_total > 0, "no goalkicks found"
    assert 0.05 <= fail_rate <= 0.60, f"goalkick fail-rate {fail_rate:.3f} outside [0.05, 0.60]"
```

- [ ] **Step 4: Confirm it collects-but-skips in CI**

Run: `python -m pytest tests/spadl/test_sportec_playeval_e2e.py -q`
Expected: `1 skipped` (no DATABRICKS_* env locally) — NOT a collection error. Proves the module
imports cleanly (incl. `import scripts._loader_databricks`, which must not pull the connector).

- [ ] **Step 5: Lint the e2e**

Run: `python -m ruff check tests/spadl/test_sportec_playeval_e2e.py && python -m ruff format --check tests/spadl/test_sportec_playeval_e2e.py`
Expected: no errors.

> **Owner-only validation (post-merge, not a CI step):** in an isolated env with the connector +
> `DATABRICKS_*`, first sanity-probe `fetch_idsse_events()` on one match (confirm `play_evaluation`,
> `team`, and `period` are populated and the column names/values match the assumptions above), then run
> `pytest tests/spadl/test_sportec_playeval_e2e.py -m e2e -s` → expect warn-silent + goalkick
> fail-rate ~0.29 (DFL goalkicks ~71% complete).

---

## Task 5: Housekeeping — TODO + version + CHANGELOG

**Files:** `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `uv.lock`

- [ ] **Step 1: Remove the completed TODO item**

In `TODO.md`, delete the `**DFL play_evaluation full vocabulary (4.20.1 follow-up).**` bullet
(lines ~33-37 under `### Blocked or Deferred`). Leave the `### Blocked or Deferred` header and the
following items (xT-GK base-rate, TF-7, …) intact. Verify with `sed -n '29,45p' TODO.md` after.

- [ ] **Step 2: Bump the version 4.21.2 → 4.21.3**

`grep -n version pyproject.toml` (confirm `4.21.2` first), then set `pyproject.toml` and
`silly_kicks/__init__.py` (`__version__`) to `4.21.3`. (4.21.2 is taken by the xT-NLL e2e;
reconcile at commit time if another release landed — `git fetch --tags && git tag | grep 4.21.3`.)

- [ ] **Step 3: Add the CHANGELOG entry**

In `CHANGELOG.md`, add above `## [4.21.2] — 2026-06-09`:

```markdown
## [4.21.3] — 2026-06-09

### Changed — sportec DFL `play_evaluation` success-allowlist (completion robustness)

Native sportec pass/set-piece completion now uses a **success-allowlist** (`fail` iff the DFL
`Evaluation` is non-empty and ∉ `{successfullyCompleted, successful}`) instead of an exact
`== "unsuccessful"` match — so any unseen reason-coded failure token (e.g. `unsuccessfulBecauseOfFoul`)
is failed by construction, and a missing/empty `play_evaluation` still maps to success (no mass-fail
on non-DFL data). Single-sourced across the main and synth-distribution sites (`_extract_play_eval` +
`_play_evaluation_is_fail` + `_warn_unexpected_play_eval`); an unexpected token is warned, not silently
classified. **Aligns the native converter with the kloppy gateway** (same success set) and is
**byte-identical on observed DFL data** (the only non-success token across the 7 IDSSE matches is
`unsuccessful`) — robustness hardening, not a re-mapping. Hyrum surface: a DFL stream carrying failure
tokens beyond `unsuccessful` would shift its fail distribution. Adds a CI-everywhere native-shape
distribution regression test and an owner-gated Databricks-bronze e2e over the 7 IDSSE matches
(`fetch_idsse_events`). No shipped-API change. (TODO 4.20.1 follow-up; refines BUG-2.)
```

- [ ] **Step 4: Re-lock + verify the bump**

Run: `uv lock && grep -rn "4\.21\.3" pyproject.toml silly_kicks/__init__.py CHANGELOG.md && python -c "import silly_kicks; print(silly_kicks.__version__)"`
Expected: `4.21.3` everywhere; the import prints `4.21.3`; `uv.lock` updates the silly-kicks pin only.

---

## Task 6: Full-suite verification + final-review

**Files:** none.

- [ ] **Step 1: Full non-e2e suite**

Run: `python -m pytest tests/ -m "not e2e and not slow" -q 2>&1 | tail -5; echo "EXIT: ${PIPESTATUS[0]}"`
Expected: baseline count + the new sportec-completion tests; 0 failures; EXIT 0. (Read the real
summary line — do not narrate from memory.)

- [ ] **Step 2: Replicate the full CI lint job (whole tree)**

Run: `python -m ruff check silly_kicks/ tests/ scripts/ && python -m ruff format --check silly_kicks/ tests/ scripts/ && python -m pyright silly_kicks/`
Expected: all clean. (Whole-tree `ruff format --check` per the 4.21.0 lesson.)

- [ ] **Step 3: Confirm e2e gating once more**

Run: `python -m pytest tests/spadl/test_sportec_playeval_e2e.py -q`
Expected: `1 skipped` — the tripwire never runs/breaks in the public path.

- [ ] **Step 4: Run `/final-review`**

Invoke the `/final-review` skill (mandatory pre-commit gate). Address any findings. Confirm C4 is
unaffected (no new container/enumeration/backend/aggregator — a converter predicate + helpers + tests).

---

## Task 7: Single commit + PR (gated on explicit approval)

**Files:** all of the above.

- [ ] **Step 1: Present the diff + commit command and HOLD for explicit approval**

Run: `git status && git --no-pager diff --stat`
Present the staged set + the exact commit command. **Do not create the sentinel or commit without an
explicit per-commit "yes"** (standing policy + the sentinel hook). Once approved (user authorizes the
sentinel at `~/.claude-git-approval`):

- [ ] **Step 2: Stage + commit (single commit, message via temp file)**

Write the commit message to a temp file (apostrophes in an inline `-m` silently truncate in git-bash —
use `-F`):

```bash
git add silly_kicks/spadl/sportec.py tests/spadl/test_sportec_completion.py \
  tests/spadl/test_sportec_playeval_e2e.py scripts/_loader_databricks.py \
  TODO.md pyproject.toml silly_kicks/__init__.py CHANGELOG.md uv.lock \
  docs/superpowers/specs/2026-06-09-sportec-play-evaluation-allowlist-design.md \
  docs/superpowers/plans/2026-06-09-sportec-play-evaluation-allowlist.md
git commit -F .git/COMMIT_SK90.txt
```

Commit subject (suggested): `fix(spadl): SK-90 sportec play_evaluation success-allowlist (kloppy-aligned) -- silly-kicks 4.21.3`.
Body (in the temp file): the success-allowlist + non-empty guard, single-sourced helpers,
byte-identical-on-observed-DFL Hyrum note, kloppy alignment, the CI regression + owner-gated bronze
e2e, the review rounds addressed. End with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

- [ ] **Step 3: Push + open the PR (body via file)**

Run: `git push -u origin pr-s90-sportec-play-evaluation-allowlist`
Then write the PR body to a temp file and `gh pr create --base main --title "..." --body-file .git/PR_SK90.md`
(same apostrophe-safety reason). Remove the temp files afterward.

- [ ] **Step 4: After CI green (user signals) — merge + tag**

Squash `--admin` merge + delete branch; then `git fetch && git checkout main && git pull`, tag
`v4.21.3` (annotated) and push (triggers `publish.yml`). Never tag before main CI is green.

---

## Self-Review notes

- **Spec coverage:** success-allowlist + non-empty guard (Task 1 helpers + both sites); single-sourced
  extraction+classification+warn (Task 1); M1 dual-site warn coverage (Task 1 Steps 4-5 both call
  `_warn_unexpected_play_eval` on their relevant rows); genuinely-new unit cases + single-source guard
  (Task 2); M3 committed native-shape distribution/warn regression (Task 3); H1 owner-gated
  Databricks-bronze e2e + `fetch_idsse_events` (Task 4); L1 4.21.3 / L2 case-sensitivity docstring /
  L-C no-`n` param / L-D fail-rate band — all present. TODO removal + version + CHANGELOG (Task 5);
  verification + `/final-review` (Task 6); single gated commit (Task 7).
- **Type/name consistency:** `_SUCCESS_EVAL` / `_KNOWN_EVAL` / `_extract_play_eval(df)` /
  `_play_evaluation_is_fail(play_eval)` / `_warn_unexpected_play_eval(play_eval)` /
  `fetch_idsse_events()` used identically across tasks. Site 1 passes `rows` (the
  `_build_raw_actions` filtered frame, length `n`); Site 2 passes `src`. `_ev` / `_convert` / `_PASS`
  / `_GOALKICK` / `_FAIL` / `_SUCCESS` reuse the existing `test_sportec_completion.py` fixtures.
- **No placeholders:** every code + command step is concrete. The only deliberately owner-deferred
  step is the post-merge manual e2e run (it needs the connector + credentials).
- **Plan-review round 2 (part-deux):** P1 — the warn + synth are now red-first (Task 1 Step 1 adds
  `test_unexpected_token_warns_and_fails` + `test_synth_goalkick_unseen_token_is_fail` before the
  implementation; the warn-silent test stays in Task 2 as a guard since it passes vacuously pre-impl).
  P2 — the now-false BUG-2 comments at BOTH sites are replaced, not left (Task 1 Steps 4-5). P3 — the
  e2e drops `from silly_kicks.tracking.utils import filter_extratime_frames` (verified to pull
  xgboost+numba+sklearn) for an inline dtype-robust period filter (Task 4 Step 3). P4 — commit/PR
  messages via temp file (`-F` / `--body-file`, Task 7). P5 — Task 0 fetches `origin/main` + asserts
  ancestry. P6 (declined) — the warn-silent assert is the correctly-scoped byte-identical proof; a
  raw-column `tokens == {unsuccessful}` assert risks false failures from shot-event `Evaluation`
  tokens (the warn fires only over pass/set-piece + synth rows), so the diagnostic print is kept but
  no raw-column assertion is added.
