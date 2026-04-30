# PR-S13: Docstring Examples coverage + CI guardrail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close TODO D-8 entirely — add `Examples` sections to every public function/class/method in silly-kicks's API surface (~50 surfaces remaining), plus add a CI guardrail (`tests/test_public_api_examples.py`) that fails on any future public symbol without an Example. Patch release 2.1.1.

**Architecture:** TDD-first: write the failing CI gate test, watch it report ~54 missing surfaces across 14 module files. Add Examples in 5 logical groups (standard SPADL → atomic mirrors → VAEP family → xT family → vaep/features helpers). Each group lands and shrinks the failure list; final group brings it to zero. Single commit per branch (per `feedback_commit_policy`). Spec: `docs/superpowers/specs/2026-04-30-docstring-examples-design.md`.

**Tech Stack:** Python 3.10+, pandas, ast (stdlib for the gate test), pytest. No new runtime dependencies. CI-pinned tooling: ruff 0.15.7, pyright 1.1.395, pandas-stubs 2.3.3.260113.

**Per-task commit policy:** NO per-task commits. Single commit at Task 13 after gates pass + user approval.

---

## File Structure

| Path | Action | Surfaces touched |
|---|---|---|
| `tests/test_public_api_examples.py` | Create | The CI gate (1 new file, ~80 lines) |
| `silly_kicks/spadl/utils.py` | Modify | add_pre_shot_gk_context, add_names, play_left_to_right, validate_spadl |
| `silly_kicks/spadl/kloppy.py` | Modify | convert_to_actions |
| `silly_kicks/spadl/sportec.py` | Modify | convert_to_actions |
| `silly_kicks/spadl/metrica.py` | Modify | convert_to_actions |
| `silly_kicks/atomic/spadl/utils.py` | Modify | add_gk_role, add_gk_distribution_metrics, add_pre_shot_gk_context, add_names, validate_atomic_spadl, play_left_to_right |
| `silly_kicks/vaep/base.py` | Modify | VAEP class + compute_features, compute_labels, fit, rate, score |
| `silly_kicks/vaep/hybrid.py` | Modify | HybridVAEP class |
| `silly_kicks/atomic/vaep/base.py` | Modify | AtomicVAEP class |
| `silly_kicks/xthreat.py` | Modify | ExpectedThreat class + fit, interpolator, rate |
| `silly_kicks/vaep/features.py` | Modify | feature_column_names, gamestates, play_left_to_right, simple, actiontype, actiontype_onehot, result, result_onehot, actiontype_result_onehot, result_onehot_prev_only, actiontype_result_onehot_prev_only, bodypart, bodypart_detailed, bodypart_onehot, bodypart_detailed_onehot, time, startlocation, endlocation, startpolar, endpolar, movement, player_possession_time, team, time_delta, space_delta, speed, goalscore, cross_zone, assist_type |
| `pyproject.toml` | Modify | Version 2.1.0 → 2.1.1 |
| `CHANGELOG.md` | Modify | New `[2.1.1]` entry |
| `TODO.md` | Modify | Close D-8; delete D-9 |

Total: 1 new test file + 11 modified source files + 3 admin files. ~500 lines added across source.

---

## Task 1: Write the failing CI guardrail test (Group F.1)

**Files:**
- Create: `tests/test_public_api_examples.py`

- [ ] **Step 1: Create the gate test file**

Write `tests/test_public_api_examples.py`:

```python
"""Enforce: every public function / class / method docstring includes an Examples section.

Closes D-8 (PR-S13). Backstops the discipline by failing CI when a future
PR adds a public symbol without an Examples section.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Files containing public API. Excludes underscore-prefixed modules,
# tests, and scripts.
_PUBLIC_MODULE_FILES = (
    "silly_kicks/spadl/utils.py",
    "silly_kicks/spadl/statsbomb.py",
    "silly_kicks/spadl/opta.py",
    "silly_kicks/spadl/wyscout.py",
    "silly_kicks/spadl/sportec.py",
    "silly_kicks/spadl/metrica.py",
    "silly_kicks/spadl/kloppy.py",
    "silly_kicks/atomic/spadl/utils.py",
    "silly_kicks/atomic/spadl/base.py",
    "silly_kicks/vaep/base.py",
    "silly_kicks/vaep/hybrid.py",
    "silly_kicks/vaep/features.py",
    "silly_kicks/atomic/vaep/base.py",
    "silly_kicks/xthreat.py",
)

# Pure-type symbols that don't fit the illustrative-example pattern.
# Adding a new entry here is a deliberate documentation-policy decision —
# the additive-only nature is a forcing function.
_SKIP_SYMBOLS = frozenset(
    {
        "BoundaryMetrics",  # TypedDict — fields are the documentation
        "CoverageMetrics",  # TypedDict
        "ConversionReport",  # TypedDict
    }
)


def _has_examples_section(docstring: str | None) -> bool:
    """True if the docstring contains a NumPy-style Examples section or doctest."""
    if not docstring:
        return False
    if "Examples\n    --------" in docstring or "Examples\n--------" in docstring:
        return True
    return ">>> " in docstring


def _walk_public_definitions(tree: ast.AST) -> list[tuple[str, int, str, ast.AST]]:
    """Yield (kind, lineno, qualified_name, node) for top-level public defs + public methods."""
    out: list[tuple[str, int, str, ast.AST]] = []
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") or node.name in _SKIP_SYMBOLS:
                continue
            out.append(("function", node.lineno, node.name, node))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_") or node.name in _SKIP_SYMBOLS:
                continue
            out.append(("class", node.lineno, node.name, node))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("_") or child.name in _SKIP_SYMBOLS:
                        continue
                    out.append(("method", child.lineno, f"{node.name}.{child.name}", child))
    return out


@pytest.mark.parametrize("file_path", _PUBLIC_MODULE_FILES)
def test_public_definitions_have_examples_section(file_path: str):
    """Every public function / class / method in *file_path* has an Examples section.

    See ``silly_kicks.spadl.add_possessions`` and ``silly_kicks.spadl.boundary_metrics``
    for canonical illustrative-style examples. Add a 3-7 line example showing typical
    usage; no doctest verification is required.
    """
    abs_path = REPO_ROOT / file_path
    assert abs_path.exists(), f"public-API module file does not exist: {file_path}"

    tree = ast.parse(abs_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    for kind, lineno, name, node in _walk_public_definitions(tree):
        doc = ast.get_docstring(node)
        if not _has_examples_section(doc):
            missing.append(f"  {file_path}:{lineno}  {kind}  {name}")

    assert not missing, (
        f"Public symbols in {file_path} missing 'Examples' section in docstring:\n"
        + "\n".join(missing)
        + "\n\nAdd a 3-7 line illustrative Examples section. See "
        "`silly_kicks.spadl.add_possessions` or `silly_kicks.spadl.boundary_metrics` "
        "for the canonical style. Pure-type symbols (TypedDict / dataclass) that don't "
        "fit the example pattern can be added to `_SKIP_SYMBOLS` in this test file — "
        "but only with a clear documentation-policy justification."
    )
```

- [ ] **Step 2: Run the gate; expect ~54 surfaces missing across multiple files**

Run: `uv run pytest tests/test_public_api_examples.py -v --tb=short`

Expected: 11 of 14 parametrized cases FAIL (each failing case lists the missing surfaces in that file). The remaining 3 cases that PASS are `silly_kicks/spadl/statsbomb.py`, `silly_kicks/spadl/opta.py`, `silly_kicks/spadl/wyscout.py` (their `convert_to_actions` already has Examples), plus `silly_kicks/atomic/spadl/base.py` (its `convert_to_atomic` already has an Example).

The aggregate failure list across all failing cases is the implementation checklist for Tasks 2-9. Save it for reference.

---

## Task 2: Examples for `silly_kicks/spadl/utils.py` (4 functions)

**Files:**
- Modify: `silly_kicks/spadl/utils.py`

- [ ] **Step 1: Add Examples to `add_pre_shot_gk_context` (line 403)**

Find the closing of the function's References section + before the function body starts. Insert after the References block, before `"""` ending the docstring:

```rst
    Examples
    --------
    Tag pre-shot goalkeeper context for downstream PSxG / xGOT modeling::

        actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
        actions = add_pre_shot_gk_context(actions, lookback_seconds=10.0)
        # Filter to shots where the defending GK was already engaged:
        engaged_shots = actions[actions["gk_was_engaged"]]
```

- [ ] **Step 2: Add Examples to `add_names` (line 1074)**

Find the docstring's closing `"""`. Insert before:

```rst
    Examples
    --------
    Append name columns for human-readable diagnostics::

        actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
        actions = add_names(actions)
        # actions now has type_name / result_name / bodypart_name string columns:
        actions[["type_name", "result_name", "bodypart_name"]].head()
```

- [ ] **Step 3: Add Examples to `play_left_to_right` (line 1104)**

```rst
    Examples
    --------
    Mirror all actions to a single direction (left-to-right per the home team)::

        actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
        ltr = play_left_to_right(actions, home_team_id=100)
        # All away-team actions now have flipped (start_x, start_y) / (end_x, end_y).
```

- [ ] **Step 4: Add Examples to `validate_spadl` (line 1285)**

```rst
    Examples
    --------
    Validate a converter's output conforms to the SPADL schema::

        actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
        validate_spadl(actions)  # raises ValueError on missing columns;
                                 # warns on dtype mismatches.
```

- [ ] **Step 5: Re-run the gate; verify `silly_kicks/spadl/utils.py` slot is now green**

Run: `uv run pytest tests/test_public_api_examples.py::test_public_definitions_have_examples_section -v -k utils.py 2>&1 | head -30`

Expected: the `silly_kicks/spadl/utils.py` parametrized case is now green. Other failing cases unchanged.

---

## Task 3: Examples for converter modules (kloppy / sportec / metrica)

**Files:**
- Modify: `silly_kicks/spadl/kloppy.py`
- Modify: `silly_kicks/spadl/sportec.py`
- Modify: `silly_kicks/spadl/metrica.py`

- [ ] **Step 1: Add Examples to `kloppy.convert_to_actions` (line 90)**

```rst
    Examples
    --------
    Convert a kloppy ``EventDataset`` (any provider supported by kloppy) to SPADL::

        import kloppy
        from silly_kicks.spadl import kloppy as sk_kloppy

        dataset = kloppy.statsbomb.load_open_data(match_id=7298)
        actions, report = sk_kloppy.convert_to_actions(dataset)
        # report.unrecognized_counts surfaces any provider events not yet mapped.
```

- [ ] **Step 2: Add Examples to `sportec.convert_to_actions` (line 447)**

```rst
    Examples
    --------
    Convert a Sportec/IDSSE bronze events DataFrame to SPADL::

        from silly_kicks.spadl import sportec

        actions, report = sportec.convert_to_actions(
            events,
            home_team_id="HOME",
            goalkeeper_ids={"DFL-OBJ-..."},  # optional supplementary GK ids
        )
        # report.coverage gives per-action-type counts; output schema is
        # SPORTEC_SPADL_COLUMNS (KLOPPY_SPADL_COLUMNS + 4 tackle qualifier columns).
```

- [ ] **Step 3: Add Examples to `metrica.convert_to_actions` (line 112)**

```rst
    Examples
    --------
    Convert a Metrica bronze events DataFrame to SPADL::

        from silly_kicks.spadl import metrica

        actions, report = metrica.convert_to_actions(
            events,
            home_team_id="HOME",
            goalkeeper_ids={"player_42", "player_99"},  # required for GK action recovery
        )
        # Metrica has no native GK markers — pass goalkeeper_ids to recover
        # keeper_pick_up / keeper_claim / synthesized GK distribution actions.
```

- [ ] **Step 4: Re-run the gate; verify the 3 converter slots are green**

Run: `uv run pytest tests/test_public_api_examples.py -v -k "kloppy or sportec or metrica" --tb=line 2>&1 | tail -10`

Expected: all 3 cases now pass.

---

## Task 4: Examples for `silly_kicks/atomic/spadl/utils.py` (6 atomic mirrors)

**Files:**
- Modify: `silly_kicks/atomic/spadl/utils.py`

- [ ] **Step 1: Add Examples to atomic `add_gk_role` (line 52)**

```rst
    Examples
    --------
    Tag GK roles after atomic conversion::

        from silly_kicks.atomic.spadl import convert_to_atomic
        from silly_kicks.atomic.spadl.utils import add_gk_role

        atomic = convert_to_atomic(actions)
        atomic = add_gk_role(atomic)
        # atomic["gk_role"] is now a categorical with 5 categories + None.
```

- [ ] **Step 2: Add Examples to atomic `add_gk_distribution_metrics` (line 492)**

```rst
    Examples
    --------
    Compute GK distribution metrics on an atomic-SPADL stream::

        from silly_kicks.atomic.spadl.utils import add_gk_distribution_metrics

        atomic = add_gk_distribution_metrics(atomic, long_threshold=60.0)
        # Filter to launches:
        launches = atomic[atomic["is_launch"]]
```

- [ ] **Step 3: Add Examples to atomic `add_pre_shot_gk_context` (line 676)**

```rst
    Examples
    --------
    Tag pre-shot defending-GK context on an atomic-SPADL stream::

        from silly_kicks.atomic.spadl.utils import add_pre_shot_gk_context

        atomic = add_pre_shot_gk_context(atomic, lookback_seconds=10.0)
        engaged_shots = atomic[atomic["gk_was_engaged"]]
```

- [ ] **Step 4: Add Examples to atomic `add_names` (line 812)**

```rst
    Examples
    --------
    Append name columns for human-readable diagnostics on atomic-SPADL::

        from silly_kicks.atomic.spadl.utils import add_names

        atomic = add_names(atomic)
        atomic[["type_name", "bodypart_name"]].head()
```

- [ ] **Step 5: Add Examples to `validate_atomic_spadl` (line 835)**

```rst
    Examples
    --------
    Validate an atomic converter's output conforms to the Atomic-SPADL schema::

        from silly_kicks.atomic.spadl.utils import validate_atomic_spadl

        validate_atomic_spadl(atomic)  # raises ValueError on missing columns
```

- [ ] **Step 6: Add Examples to atomic `play_left_to_right` (line 877)**

```rst
    Examples
    --------
    Mirror atomic actions to a single direction (left-to-right per the home team)::

        from silly_kicks.atomic.spadl.utils import play_left_to_right

        ltr = play_left_to_right(atomic, home_team_id=100)
        # All away-team actions now have flipped (x, y) and (dx, dy).
```

- [ ] **Step 7: Re-run gate; verify atomic/spadl/utils.py slot is green**

Run: `uv run pytest tests/test_public_api_examples.py -v -k "atomic/spadl" --tb=line`

Expected: pass.

---

## Task 5: Examples for `silly_kicks/vaep/base.py` (VAEP class + 5 methods)

**Files:**
- Modify: `silly_kicks/vaep/base.py`

The VAEP class docstring shows the full lifecycle. Each method then shows its specific call shape.

- [ ] **Step 1: Add Examples to `VAEP` class (line 44)**

Find the docstring's closing `"""` (after the References section). Insert before:

```rst
    Examples
    --------
    Train a VAEP model and rate actions for a single game::

        import pandas as pd
        from silly_kicks.vaep import VAEP

        v = VAEP()
        # Compute features + labels across many games:
        X_list, y_list = [], []
        for _, game in games.iterrows():
            game_actions = actions[actions["game_id"] == game.game_id]
            X_list.append(v.compute_features(game, game_actions))
            y_list.append(v.compute_labels(game, game_actions))
        X, y = pd.concat(X_list), pd.concat(y_list)
        v.fit(X, y, learner="xgboost")

        # Rate one game's actions:
        ratings = v.rate(game, game_actions)
        # ratings has columns: offensive_value / defensive_value / vaep_value
```

- [ ] **Step 2: Add Examples to `VAEP.compute_features` (line 93)**

Insert into the existing docstring before the closing `"""`:

```rst
        Examples
        --------
        Compute the feature representation for one game::

            X = v.compute_features(game, game_actions)
            # X has one row per game state with the columns specified by ``v.xfns``.
```

- [ ] **Step 3: Add Examples to `VAEP.compute_labels` (line 114)**

```rst
        Examples
        --------
        Compute the label representation (scores / concedes binaries) for one game::

            y = v.compute_labels(game, game_actions)
            # y has columns: scores / concedes (next-N-action lookahead).
```

- [ ] **Step 4: Add Examples to `VAEP.fit` (line 137)**

```rst
        Examples
        --------
        Fit a VAEP model with xgboost (default) on accumulated features + labels::

            v.fit(X, y, learner="xgboost", val_size=0.25)
            # ``random_state`` controls the train/val split deterministically.
```

- [ ] **Step 5: Add Examples to `VAEP.rate` (line 217)**

```rst
        Examples
        --------
        Rate one game's actions after fitting::

            ratings = v.rate(game, game_actions)
            ratings[["offensive_value", "defensive_value", "vaep_value"]].head()
```

- [ ] **Step 6: Add Examples to `VAEP.score` (line 259)**

```rst
        Examples
        --------
        Evaluate fit quality on held-out data::

            metrics = v.score(X_test, y_test)
            # metrics["scores"]["brier"] / metrics["scores"]["roc_auc"], same for concedes.
```

- [ ] **Step 7: Re-run gate; verify vaep/base.py slot is green**

Run: `uv run pytest tests/test_public_api_examples.py -v -k vaep/base --tb=line`

Expected: pass.

---

## Task 6: Examples for `silly_kicks/vaep/hybrid.py` (HybridVAEP class)

**Files:**
- Modify: `silly_kicks/vaep/hybrid.py`

- [ ] **Step 1: Add Examples to `HybridVAEP` class (line 37)**

Insert before the closing `"""`:

```rst
    Examples
    --------
    Train a HybridVAEP model (result leakage removed from current action)::

        from silly_kicks.vaep.hybrid import HybridVAEP

        v = HybridVAEP()
        # Same compute_features / compute_labels / fit / rate lifecycle as VAEP.
        # The only difference is the default xfns list — see ``hybrid_xfns_default``.
        v.fit(X, y)
        ratings = v.rate(game, game_actions)
```

- [ ] **Step 2: Re-run gate; verify hybrid.py slot is green**

Run: `uv run pytest tests/test_public_api_examples.py -v -k hybrid --tb=line`

Expected: pass.

---

## Task 7: Examples for `silly_kicks/atomic/vaep/base.py` (AtomicVAEP class)

**Files:**
- Modify: `silly_kicks/atomic/vaep/base.py`

- [ ] **Step 1: Add Examples to `AtomicVAEP` class (line 35)**

Insert before the closing `"""` (after References block):

```rst
    Examples
    --------
    Train an AtomicVAEP model on an atomic-SPADL stream::

        from silly_kicks.atomic.vaep import AtomicVAEP

        v = AtomicVAEP()
        # Compute features + labels per game on the atomic stream:
        X_list, y_list = [], []
        for _, game in games.iterrows():
            game_atomic = atomic[atomic["game_id"] == game.game_id]
            X_list.append(v.compute_features(game, game_atomic))
            y_list.append(v.compute_labels(game, game_atomic))
        X, y = pd.concat(X_list), pd.concat(y_list)
        v.fit(X, y)
        ratings = v.rate(game, game_atomic)
```

- [ ] **Step 2: Re-run gate; verify atomic/vaep/base.py slot is green**

Run: `uv run pytest tests/test_public_api_examples.py -v -k "atomic/vaep" --tb=line`

Expected: pass.

---

## Task 8: Examples for `silly_kicks/xthreat.py` (ExpectedThreat class + 3 methods)

**Files:**
- Modify: `silly_kicks/xthreat.py`

- [ ] **Step 1: Add Examples to `ExpectedThreat` class (line 211)**

Insert before the closing `"""` (after References block):

```rst
    Examples
    --------
    Fit an Expected Threat (xT) grid and rate actions::

        from silly_kicks.xthreat import ExpectedThreat

        xt = ExpectedThreat()
        xt.fit(actions)
        values = xt.rate(actions)  # ndarray of shape (len(actions),)
```

- [ ] **Step 2: Add Examples to `ExpectedThreat.fit` (line 302)**

```rst
        Examples
        --------
        Fit the xT grid on a SPADL action stream::

            xt = ExpectedThreat().fit(actions)
            # xt.xT is the (W, L) value surface; xt.heatmaps records each iteration.
```

- [ ] **Step 3: Add Examples to `ExpectedThreat.interpolator` (line 328)**

```rst
        Examples
        --------
        Interpolate xT values across continuous coordinates::

            interp = xt.interpolator(kind="linear")
            grid = interp(xs, ys)  # (len(ys), len(xs)) array — y on first axis.
```

- [ ] **Step 4: Add Examples to `ExpectedThreat.rate` (line 381)**

```rst
        Examples
        --------
        Rate move-class actions in a SPADL stream::

            xt = ExpectedThreat().fit(actions)
            values = xt.rate(actions, use_interpolation=True)
            # Non-move actions (shots / fouls / etc.) receive NaN.
```

- [ ] **Step 5: Re-run gate; verify xthreat.py slot is green**

Run: `uv run pytest tests/test_public_api_examples.py -v -k xthreat --tb=line`

Expected: pass.

---

## Task 9: Examples for `silly_kicks/vaep/features.py` (29 helpers)

**Files:**
- Modify: `silly_kicks/vaep/features.py`

Largest task by surface count. All examples share the preamble pattern: `states = gamestates(actions, nb_prev_actions=3); feats = <fn>(states)`.

For brevity, the steps below group examples that follow the same template. Each step writes the examples for a logical sub-group within the file.

- [ ] **Step 1: Add Examples to `feature_column_names` (line 18) and `gamestates` (line 60)**

Insert into `feature_column_names` docstring before the closing `"""`:

```rst
    Examples
    --------
    Discover the feature column names a feature-fn list will produce::

        from silly_kicks.vaep import features as fs

        cols = fs.feature_column_names([fs.actiontype_onehot, fs.bodypart_onehot])
        # cols includes the cartesian product of feature × previous-action index.
```

Insert into `gamestates` docstring before the closing `"""`:

```rst
    Examples
    --------
    Build a 3-step gamestate stream from a SPADL action DataFrame::

        from silly_kicks.vaep.features import gamestates

        states = gamestates(actions, nb_prev_actions=3)
        # ``states`` is a list of 3 DataFrames — states[0] is the current action,
        # states[1] is the previous action aligned by row, states[2] is the one before.
```

- [ ] **Step 2: Add Examples to `play_left_to_right` (line 109)**

```rst
    Examples
    --------
    Mirror gamestates to a single direction (per home_team_id)::

        from silly_kicks.vaep.features import play_left_to_right

        ltr_states = play_left_to_right(states, home_team_id=100)
        # Every away-team gamestate now has flipped (start_x, start_y) / (end_x, end_y).
```

- [ ] **Step 3: Add Examples to `simple` (line 143)**

```rst
    Examples
    --------
    Compute the canonical "simple" feature vector for each gamestate::

        from silly_kicks.vaep.features import simple

        feats = simple(states)
```

- [ ] **Step 4: Add Examples to `actiontype` (line 202) and `actiontype_onehot` (line 219)**

Both follow the same template. For `actiontype`:

```rst
    Examples
    --------
    Extract action-type integer codes per gamestate slot::

        from silly_kicks.vaep.features import actiontype

        feats = actiontype(states)
```

For `actiontype_onehot`:

```rst
    Examples
    --------
    Extract action-type one-hot features per gamestate slot::

        from silly_kicks.vaep.features import actiontype_onehot

        feats = actiontype_onehot(states)
```

- [ ] **Step 5: Add Examples to `result` (line 240) and `result_onehot` (line 263)**

For `result`:

```rst
    Examples
    --------
    Extract action-result integer codes per gamestate slot::

        from silly_kicks.vaep.features import result

        feats = result(states)
```

For `result_onehot`:

```rst
    Examples
    --------
    Extract action-result one-hot features per gamestate slot::

        from silly_kicks.vaep.features import result_onehot

        feats = result_onehot(states)
```

- [ ] **Step 6: Add Examples to `actiontype_result_onehot` (line 284)**

```rst
    Examples
    --------
    Extract joint action-type × result one-hot features per gamestate slot::

        from silly_kicks.vaep.features import actiontype_result_onehot

        feats = actiontype_result_onehot(states)
```

- [ ] **Step 7: Add Examples to `result_onehot_prev_only` (line 304) and `actiontype_result_onehot_prev_only` (line 329)**

For `result_onehot_prev_only`:

```rst
    Examples
    --------
    Result one-hot features for *previous* actions only (HybridVAEP — removes
    result leakage on the current action)::

        from silly_kicks.vaep.features import result_onehot_prev_only

        feats = result_onehot_prev_only(states)
        # feats has columns for a1, a2, ... but not a0.
```

For `actiontype_result_onehot_prev_only`:

```rst
    Examples
    --------
    Joint action-type × result one-hot for *previous* actions only::

        from silly_kicks.vaep.features import actiontype_result_onehot_prev_only

        feats = actiontype_result_onehot_prev_only(states)
```

- [ ] **Step 8: Add Examples to `bodypart` / `bodypart_detailed` / `bodypart_onehot` / `bodypart_detailed_onehot` (lines 355, 390, 421, 462)**

For `bodypart`:

```rst
    Examples
    --------
    Extract bodypart integer codes per gamestate slot::

        from silly_kicks.vaep.features import bodypart

        feats = bodypart(states)
```

For `bodypart_detailed`:

```rst
    Examples
    --------
    Extract detailed bodypart integer codes (foot_left / foot_right resolved)::

        from silly_kicks.vaep.features import bodypart_detailed

        feats = bodypart_detailed(states)
```

For `bodypart_onehot`:

```rst
    Examples
    --------
    Extract bodypart one-hot features per gamestate slot::

        from silly_kicks.vaep.features import bodypart_onehot

        feats = bodypart_onehot(states)
```

For `bodypart_detailed_onehot`:

```rst
    Examples
    --------
    Extract detailed bodypart one-hot features per gamestate slot::

        from silly_kicks.vaep.features import bodypart_detailed_onehot

        feats = bodypart_detailed_onehot(states)
```

- [ ] **Step 9: Add Examples to `time` (line 502)**

```rst
    Examples
    --------
    Extract clock-time features per gamestate slot::

        from silly_kicks.vaep.features import time

        feats = time(states)
        # Columns: time_seconds_overall_a0, time_seconds_a0, etc.
```

- [ ] **Step 10: Add Examples to spatial helpers — `startlocation` / `endlocation` / `startpolar` / `endpolar` (lines 532, 549, 566, 591)**

For `startlocation`:

```rst
    Examples
    --------
    Extract action start-location features per gamestate slot::

        from silly_kicks.vaep.features import startlocation

        feats = startlocation(states)
```

For `endlocation`:

```rst
    Examples
    --------
    Extract action end-location features per gamestate slot::

        from silly_kicks.vaep.features import endlocation

        feats = endlocation(states)
```

For `startpolar`:

```rst
    Examples
    --------
    Extract polar (distance/angle to goal) start features per gamestate slot::

        from silly_kicks.vaep.features import startpolar

        feats = startpolar(states)
```

For `endpolar`:

```rst
    Examples
    --------
    Extract polar (distance/angle to goal) end features per gamestate slot::

        from silly_kicks.vaep.features import endpolar

        feats = endpolar(states)
```

- [ ] **Step 11: Add Examples to `movement` (line 616) and `player_possession_time` (line 638)**

For `movement`:

```rst
    Examples
    --------
    Extract action displacement (dx, dy, total) features per gamestate slot::

        from silly_kicks.vaep.features import movement

        feats = movement(states)
```

For `player_possession_time`:

```rst
    Examples
    --------
    Extract per-action player-possession duration (seconds) features::

        from silly_kicks.vaep.features import player_possession_time

        feats = player_possession_time(states)
```

- [ ] **Step 12: Add Examples to `team` / `time_delta` / `space_delta` / `speed` (lines 668, 693, 714, 740)**

For `team`:

```rst
    Examples
    --------
    Extract team-of-actor features per gamestate slot::

        from silly_kicks.vaep.features import team

        feats = team(states)
```

For `time_delta`:

```rst
    Examples
    --------
    Extract time-since-previous-action features per gamestate slot::

        from silly_kicks.vaep.features import time_delta

        feats = time_delta(states)
```

For `space_delta`:

```rst
    Examples
    --------
    Extract space-since-previous-action features per gamestate slot::

        from silly_kicks.vaep.features import space_delta

        feats = space_delta(states)
```

For `speed`:

```rst
    Examples
    --------
    Extract speed (space_delta / time_delta) features per gamestate slot::

        from silly_kicks.vaep.features import speed

        feats = speed(states)
```

- [ ] **Step 13: Add Examples to `goalscore` / `cross_zone` / `assist_type` (lines 771, 805, 851)**

For `goalscore`:

```rst
    Examples
    --------
    Extract per-action goal-difference features (own goals scored vs conceded)::

        from silly_kicks.vaep.features import goalscore

        feats = goalscore(states)
```

For `cross_zone`:

```rst
    Examples
    --------
    Extract cross-zone categorical features per gamestate slot::

        from silly_kicks.vaep.features import cross_zone

        feats = cross_zone(states)
```

For `assist_type`:

```rst
    Examples
    --------
    Extract assist-type categorical features per gamestate slot::

        from silly_kicks.vaep.features import assist_type

        feats = assist_type(states)
```

- [ ] **Step 14: Re-run gate; verify vaep/features.py slot is green**

Run: `uv run pytest tests/test_public_api_examples.py -v -k features --tb=line`

Expected: pass.

- [ ] **Step 15: Run the full gate one more time; expect ALL slots green**

Run: `uv run pytest tests/test_public_api_examples.py -v --tb=line`

Expected: 14 passed.

---

## Task 10: Verification gates

**Files:** none modified — verification only

- [ ] **Step 1: Pin tooling to CI versions**

Run: `uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113`

Expected: clean install message.

- [ ] **Step 2: Lint check**

Run: `uv run ruff check silly_kicks/ tests/`

Expected: zero errors.

- [ ] **Step 3: Format check**

Run: `uv run ruff format --check silly_kicks/ tests/`

Expected: zero changes needed. If formatting issues, run `uv run ruff format silly_kicks/ tests/` and re-check.

- [ ] **Step 4: Type check**

Run: `uv run pyright silly_kicks/`

Expected: zero errors.

- [ ] **Step 5: Full pytest suite (background, ~30s)**

Run: `uv run pytest tests/ --tb=short` with `run_in_background=true` (full suite exceeds 30s timeout).

Expected: 788 passed, 4 skipped, 0 failed (774 baseline + 14 new parametrized cases).

- [ ] **Step 6: Spot-check rendering on one example per group**

Run: `uv run python -c "from silly_kicks.spadl.utils import add_names; help(add_names)" | grep -A 5 Examples`

Run: `uv run python -c "from silly_kicks.atomic.spadl.utils import add_gk_role; help(add_gk_role)" | grep -A 5 Examples`

Run: `uv run python -c "from silly_kicks.vaep import VAEP; help(VAEP)" | grep -A 8 Examples`

Run: `uv run python -c "from silly_kicks.xthreat import ExpectedThreat; help(ExpectedThreat)" | grep -A 5 Examples`

Run: `uv run python -c "from silly_kicks.vaep.features import simple; help(simple)" | grep -A 5 Examples`

Expected: each command shows the Examples section rendering correctly (proper indentation, no Sphinx-directive parse errors visible).

---

## Task 11: /final-review

- [ ] **Step 1: Invoke /final-review**

Use the Skill tool to launch `mad-scientist-skills:final-review`.

Expected: skill produces a quality report. Address any actionable findings inline.

- [ ] **Step 2: Address findings**

If `/final-review` flags any issues (style inconsistency, missing examples, etc.), fix them and re-run until clean.

---

## Task 12: CHANGELOG, TODO, version bump

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `TODO.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Insert `[2.1.1]` entry at the top of CHANGELOG.md (after the header, before `[2.1.0]`)**

```markdown
## [2.1.1] — 2026-04-30

### Added

- **Examples sections on all public API surfaces.** Closes the long-standing D-8
  documentation gap. Every public function / class / method in
  `silly_kicks.spadl`, `silly_kicks.atomic.spadl`, `silly_kicks.vaep`,
  `silly_kicks.atomic.vaep`, and `silly_kicks.xthreat` now has a 3-7 line
  illustrative example showing typical usage. ~50 surfaces newly documented.
- **CI guardrail at `tests/test_public_api_examples.py`.** AST-based parametrized
  test asserts every public symbol has an `Examples` section in its docstring.
  Future PRs that add a public function without an Example fail CI; the failure
  message points to canonical-style references (`add_possessions`,
  `boundary_metrics`).

### Changed

- **D-9 entry removed from `TODO.md`.** Tech-debt entry was stale — all 9
  module-level helpers in `silly_kicks/xthreat.py` are already underscore-
  prefixed; the entry tracked work that was completed prior to silly-kicks 2.0.0.

No API or behavior changes.
```

- [ ] **Step 2: Close D-8 in TODO.md (move out of "Documentation" table)**

Find:

```markdown
## Documentation

| # | Size | Item | Context |
|---|------|------|---------|
| D-8 | Large | Add docstring `Examples` sections to public functions | 49 public functions, zero examples. Start with the 10 most-used: `convert_to_actions` (×4 providers), `VAEP.fit`, `VAEP.rate`, `gamestates`, `add_names`, `validate_spadl`, `HybridVAEP` |
```

Replace with:

```markdown
## Documentation

(none currently queued — D-8 closed in silly-kicks 2.1.1)
```

- [ ] **Step 3: Delete D-9 from TODO.md "Tech Debt" table**

Find:

```markdown
| D-9 | Low | 5 xthreat module-level functions (`scoring_prob`, `get_move_actions`, etc.) not underscore-prefixed but not re-exported | Implementation helpers technically public API. Audit-source: DEFERRED.md (migrated 1.9.0). |
```

Delete this entire row.

- [ ] **Step 4: Bump pyproject.toml version**

Find: `version = "2.1.0"`
Replace: `version = "2.1.1"`

---

## Task 13: Single-commit gate (user approval required)

Per `feedback_commit_policy`. Branch should already be `feat/docstring-examples-coverage` (created at the start of execution).

- [ ] **Step 1: Confirm branch state + diff summary**

Run: `git status` and `git diff main --stat`

Expected: branch points to feat/docstring-examples-coverage, working tree shows the expected files modified per the File Structure table, no extraneous changes.

- [ ] **Step 2: Wait for explicit user approval**

Show the user the diff summary + the prepared commit message. Do NOT commit without approval.

- [ ] **Step 3: Single commit**

```bash
git add tests/test_public_api_examples.py \
    silly_kicks/spadl/utils.py \
    silly_kicks/spadl/kloppy.py \
    silly_kicks/spadl/sportec.py \
    silly_kicks/spadl/metrica.py \
    silly_kicks/atomic/spadl/utils.py \
    silly_kicks/vaep/base.py \
    silly_kicks/vaep/hybrid.py \
    silly_kicks/atomic/vaep/base.py \
    silly_kicks/xthreat.py \
    silly_kicks/vaep/features.py \
    pyproject.toml \
    CHANGELOG.md \
    TODO.md \
    docs/superpowers/specs/2026-04-30-docstring-examples-design.md \
    docs/superpowers/plans/2026-04-30-docstring-examples.md

git commit -s -m "$(cat <<'EOF'
docs: Examples sections on all public API surfaces + CI guardrail -- silly-kicks 2.1.1

PR-S13. Closes long-standing TODO D-8 by adding 3-7 line illustrative Examples
sections to every public function / class / method in silly-kicks's API
(~50 surfaces newly documented across spadl, atomic.spadl, vaep, atomic.vaep,
xthreat). Style matches the established PR-S8/S12 pattern (add_possessions,
boundary_metrics as canonical references).

New CI guardrail at tests/test_public_api_examples.py — AST-based parametrized
test asserts every public symbol has an Examples section. Future PRs that add
a public function without an Example fail CI; the failure message points to
the canonical-style references. Three TypedDict pure-type symbols are
explicitly allowlisted (BoundaryMetrics, CoverageMetrics, ConversionReport)
where examples don't fit the data-shape-as-docs pattern.

D-9 closed (TODO entry was stale — all xthreat module-level helpers are
already underscore-prefixed; ExpectedThreat is the only public top-level
symbol; entry tracked work completed prior to 2.0.0).

Patch release. No API or behavior changes.

Spec: docs/superpowers/specs/2026-04-30-docstring-examples-design.md
Plan: docs/superpowers/plans/2026-04-30-docstring-examples.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify the commit**

Run: `git log --oneline -2 && git diff HEAD^ HEAD --stat | tail -20`

Expected: one new commit on the branch; diff stats match the File Structure table.

---

## Task 14: Push, PR, merge, tag (user approval required at each step)

- [ ] **Step 1: User approval to push**

Wait for explicit approval.

- [ ] **Step 2: Push branch**

Run: `git push -u origin feat/docstring-examples-coverage`

- [ ] **Step 3: User approval to open PR**

Wait for approval.

- [ ] **Step 4: Open PR**

```bash
gh pr create --title "docs: Examples sections on all public API surfaces + CI guardrail -- silly-kicks 2.1.1" --body "$(cat <<'EOF'
## Summary

PR-S13. Closes the long-standing D-8 documentation gap: every public function / class / method in silly-kicks's API now has a 3-7 line illustrative `Examples` section in its docstring (~50 surfaces newly documented across `spadl`, `atomic.spadl`, `vaep`, `atomic.vaep`, `xthreat`). Style matches the established PR-S8/S12 pattern.

New CI guardrail at `tests/test_public_api_examples.py` — AST-based parametrized test asserts every public symbol has an `Examples` section. Future PRs that add a public function without an Example fail CI; the failure message points to canonical-style references (`add_possessions`, `boundary_metrics`). Three TypedDict pure-type symbols are explicitly allowlisted.

D-9 also closed (TODO entry was stale).

Patch release. No API or behavior changes.

## Test plan

- [x] Full local pytest suite (788 passed, 4 skipped, 0 failed — was 774 baseline + 14 new parametrized cases)
- [x] ruff check + format clean (CI-pinned 0.15.7)
- [x] pyright clean (CI-pinned 1.1.395 + pandas-stubs 2.3.3.260113)
- [x] Spot-check `help(...)` rendering on one symbol per group
- [x] `/final-review` clean
- [ ] CI matrix green on all platforms (3.10/3.11/3.12 ubuntu + 3.12 windows)

## Spec / Plan

- Spec: `docs/superpowers/specs/2026-04-30-docstring-examples-design.md`
- Plan: `docs/superpowers/plans/2026-04-30-docstring-examples.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: User approval to merge**

Wait for approval. Confirm CI is green before merging.

- [ ] **Step 6: Merge (squash + delete branch)**

Run: `gh pr merge --admin --squash --delete-branch`

- [ ] **Step 7: User approval to tag**

Wait for approval.

- [ ] **Step 8: Switch to main, pull, tag, push**

```bash
git checkout main
git pull
git tag v2.1.1
git push origin v2.1.1
```

Expected: tag push fires the PyPI auto-publish workflow.

- [ ] **Step 9: Verify PyPI release**

Run: `gh run list --limit 2 | head -3`

Expected: a "Publish to PyPI" run for `v2.1.1` queued / in progress / completed-success.

---

## Self-Review Checklist

Spec coverage check (against `docs/superpowers/specs/2026-04-30-docstring-examples-design.md`):

- §2 Goal 1 (close D-8 entirely) → Tasks 2-9 (one task per file with all surfaces in that file)
- §2 Goal 2 (CI guardrail) → Task 1 (gate written first); Tasks 2-9 verify slot-by-slot; Task 9 final aggregate verification
- §2 Goal 3 (D-9 close-out) → Task 12 step 3 (delete the row)
- §2 Goal 4 (patch release 2.1.1) → Task 12 step 4 (version bump); Task 14 (tag)

§4.1 File structure → Task table at the top of this plan matches §4.1 of the spec.

§4.2 Style — illustrative → All Examples in Tasks 2-9 are illustrative (3-7 lines, context-free, no doctest verification). Match canonical references.

§4.3 Per-group templates → Tasks 2-9 follow the per-group templates from §4.3 of the spec.

§4.4 CI guardrail code → Task 1 step 1 contains the literal test file content from §4.4 of the spec.

§4.5 D-9 close-out → Task 12 step 3 deletes the stale TODO row.

§5 Implementation order → Tasks 1-9 follow the TDD-first ordering from §5 of the spec (gate first, then groups A-E shrink the failure list).

§6 Verification gates → Task 10 (lint, format, typecheck, pytest) + Task 11 (/final-review).

§7 Commit cycle → Tasks 13-14 (single commit gate + push/PR/merge/tag with user approvals).

§9 Acceptance criteria — all 10 mapped:
1. Every public surface has Examples → Tasks 2-9 (verified by gate test passing in Task 9 step 15)
2. `tests/test_public_api_examples.py` exists, passes, serves as gate → Task 1
3. Style consistent → Tasks 2-9 follow §4.3 templates
4. TODO D-8 closed, D-9 deleted → Task 12 steps 2-3
5. CHANGELOG `[2.1.1]` entry → Task 12 step 1
6. pyproject version 2.1.1 → Task 12 step 4
7. Verification gates pass → Tasks 10-11
8. Tag `v2.1.1` pushed → PyPI fires → Task 14 steps 8-9
9. Existing examples unchanged → No task touches existing examples (only adds new ones)
10. Total test count 788 → Task 10 step 5 expectation

Type-consistency check:
- `actions` (SPADL DataFrame), `atomic` (atomic-SPADL DataFrame), `states` (gamestates output), `feats` (feature DataFrame), `v` (VAEP/HybridVAEP/AtomicVAEP instance), `xt` (ExpectedThreat instance) — variable naming consistent across all 50+ examples ✓
- `convert_to_actions(events, home_team_id=...)` signature — matches actual signatures across 6 providers ✓
- `gamestates(actions, nb_prev_actions=3)` — matches actual signature ✓
- `VAEP().fit(X, y, ...)` — matches actual signature (X, y are pre-computed feature/label DataFrames per game accumulation) ✓
- `xt.fit(actions)` and `xt.rate(actions)` — match actual signatures ✓

Placeholder scan: no "TBD", "TODO", "implement later", "fill in details", "add appropriate error handling", or "similar to Task N" in any task. All steps contain literal code or exact commands.
