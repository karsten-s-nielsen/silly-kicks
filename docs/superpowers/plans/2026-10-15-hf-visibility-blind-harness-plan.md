# Visibility-blind HF training harness — Implementation Plan

> **For workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan
> task-by-task, INLINE in the session. **Subagents are banned on this project — do NOT use
> `subagent-driven-development`.** Worktrees are banned — work on the feature branch directly. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the HF in-job trainer code (`xshot_occurrence`, `xcross_attempt`) visibility-blind — it
receives the public/owner arm membership as an explicit neutral job parameter and never encodes a
public/private rule — so the agnostic `--data-dir` path reproduces the committed visibility-gated
`public` candidate faithfully.

**Architecture:** Policy (which matches are public) is computed by the launcher off-HF from the in-repo
`_corpus.PUBLIC_CORPUS`; it is passed to the job as a neutral `--arms` JSON of dataset match-keys →
candidate-arm memberships. The in-job trainer builds the candidate masks by `np.isin` membership, with no
`_corpus`/`_loader_pining` visibility import and no provider-name logic. The pining `--providers` path is
removed from the trainers entirely (the maintainer/DGX pining workflow adopts the launcher + `--arms`).

**Tech Stack:** Python 3.10 (`.venv`) / 3.12 (CI, DGX), numpy, pandas, pytest; the HF Jobs harness
(`scripts/hf_jobs/`), the trainers (`scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py`),
the shared study-inputs cache (`scripts/_study_shared.py`).

**Spec:** `docs/superpowers/specs/2026-10-15-hf-visibility-blind-harness-design.md` (ADR-123 Amendment B).
Executors read both. Review CONSIDERs folded: HFVB-SPEC-01 (transitive I1 gate — Task 6), HFVB-SPEC-02
(smoke-path `--arms` note — Task 7), HVB-SPEC-01 (arm keys are DATASET keys; arm⊆corpus test — Task 5).

## Global Constraints

Copied verbatim from the spec; every task's requirements implicitly include these.

- **I1 — in-job code is visibility-blind.** No module reachable from an HF job (`scripts/train_<visibility
  trainer>.py` + its transitive runtime imports) may import or reference `_corpus.PUBLIC_CORPUS`,
  `_corpus.is_public_row`, `_corpus.assert_public_corpus`, `_corpus.artifact_label`, or
  `_loader_pining.match_visibility`. The public/owner taxonomy is launcher/off-HF only.
- **I2 — policy enters as an explicit job parameter.** A neutral `--arms PATH` JSON of dataset match-keys
  → arm memberships; the job applies it mechanically, never recomputes it.
- **I3 — datasets stay purely neutral.** No visibility column in the staged parquet. Visibility is a
  per-run parameter, not dataset content.
- **I4 — fail-loud.** A visibility-gated trainer invoked with no `--arms` raises an explicit error — never
  a silent all-private fallback.
- **Arm keys are the DATASET's own keys** (`<source_provider>__<match_id>`, e.g. `sportec__DFL-MAT-J03WMX`);
  the launcher maps `sportec`→`idsse` INTERNALLY only for the `PUBLIC_CORPUS` lookup and EMITS dataset keys.
- **Scope:** `xshot_occurrence` + `xcross_attempt` only. `ghost_gk`, `ghost_outfield`, `receiver_model`,
  and the four public trainers ship no visibility-gated candidate and are not touched.
- **Commit discipline (owner, overrides the skill):** NO per-task commits and NO per-step commits. The
  whole change is ONE commit, bundled with the weights commit, made only on Karsten's explicit go, and it
  does NOT amend the pushed `6f6508a`/`ac875c4`. Each task ends with its tests green, not a commit.
- **Lint/type scope:** `python -m ruff check silly_kicks/ tests/ scripts/` + `--format --check`; `pyright`
  scoped to changed files. Full suite under `.venv\Scripts\python.exe` (py3.10.19), CI authoritative.
- **No subagents, no worktrees.** Execute inline on `feat/hf-jobs-training-harness`.

---

### Task 1: Arm-spec primitive (pure)

A pure module that parses the `--arms` JSON and builds per-row masks by membership — the single source of
truth both trainers and the tests use. No numpy policy, no provider names.

**Files:**
- Create: `scripts/hf_jobs/arms.py`
- Test: `tests/scripts/hf_jobs/test_arms.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class ArmSpec: arms: dict[str, frozenset[str]]; ship_order: tuple[str, ...]`
  - `load_arms(path: str | Path) -> ArmSpec` — reads the JSON; raises `ArmsUnavailableError` if the file is
    missing or has no non-empty arm.
  - `arm_masks(match_keys: Sequence[str], spec: ArmSpec) -> dict[str, np.ndarray]` — `{name:
    np.isin(match_keys, sorted(members))}` for each arm in `spec.arms`.
  - `class ArmsUnavailableError(RuntimeError)`
  - `match_key(provider: str, match_id: str) -> str` → `f"{provider}__{match_id}"` (the dataset key, the
    SAME composite the trainers' `for_each`/`_iter_matches_from_dir` and the shard filenames use).

- [ ] **Step 1: Write the failing tests**

```python
# tests/scripts/hf_jobs/test_arms.py
import json
import numpy as np
import pytest
from scripts.hf_jobs.arms import ArmSpec, ArmsUnavailableError, arm_masks, load_arms, match_key


def test_match_key_is_provider_double_underscore_id():
    assert match_key("sportec", "DFL-MAT-J03WMX") == "sportec__DFL-MAT-J03WMX"


def test_load_arms_round_trips(tmp_path):
    p = tmp_path / "arms.json"
    p.write_text(json.dumps({
        "arms": {"public": ["sportec__DFL-MAT-J03WMX", "skillcorner__1874553"],
                 "sc_extended": ["sportec__DFL-MAT-J03WMX", "skillcorner__1874553", "skillcorner__9999"]},
        "ship_order": ["public", "sc_extended"],
    }))
    spec = load_arms(p)
    assert spec.ship_order == ("public", "sc_extended")
    assert spec.arms["public"] == frozenset({"sportec__DFL-MAT-J03WMX", "skillcorner__1874553"})


def test_arm_masks_membership_by_dataset_key():
    keys = ["sportec__DFL-MAT-J03WMX", "skillcorner__1874553", "skillcorner__9999"]
    spec = ArmSpec(arms={"public": frozenset(keys[:2]), "full": frozenset(keys)},
                   ship_order=("public", "full"))
    m = arm_masks(keys, spec)
    assert m["public"].tolist() == [True, True, False]
    assert m["full"].tolist() == [True, True, True]


def test_load_arms_missing_file_raises(tmp_path):
    with pytest.raises(ArmsUnavailableError):
        load_arms(tmp_path / "nope.json")


def test_load_arms_all_empty_raises(tmp_path):
    p = tmp_path / "arms.json"
    p.write_text(json.dumps({"arms": {"public": []}, "ship_order": ["public"]}))
    with pytest.raises(ArmsUnavailableError):
        load_arms(p)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/hf_jobs/test_arms.py -v`
Expected: FAIL (`ModuleNotFoundError: scripts.hf_jobs.arms`).

- [ ] **Step 3: Implement `scripts/hf_jobs/arms.py`**

```python
"""Neutral candidate-arm membership, parsed from the --arms job parameter (ADR-123 Amendment B).

The in-job trainers build their candidate row masks from THIS alone -- a mapping of dataset match-keys to
arm memberships, supplied by the launcher. There is NO public/private rule here and NO provider-name
logic: an arm is just a set of keys. Policy (which keys are `public`) lives in the launcher, off-HF.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ArmsUnavailableError(RuntimeError):
    """The --arms parameter is absent or defines no non-empty arm (I4: fail-loud, never all-private)."""


@dataclass(frozen=True)
class ArmSpec:
    arms: dict[str, frozenset[str]]
    ship_order: tuple[str, ...]


def match_key(provider: str, match_id: str) -> str:
    """The dataset's own match key: `<source_provider>__<match_id>` (shard-filename / for_each key)."""
    return f"{provider}__{match_id}"


def load_arms(path: str | Path) -> ArmSpec:
    p = Path(path)
    if not p.exists():
        raise ArmsUnavailableError(f"--arms file not found: {p}")
    raw = json.loads(p.read_text())
    arms = {name: frozenset(map(str, members)) for name, members in raw.get("arms", {}).items()}
    if not any(arms.values()):
        raise ArmsUnavailableError(f"--arms defines no non-empty arm: {p}")
    ship_order = tuple(raw.get("ship_order") or sorted(arms))
    return ArmSpec(arms=arms, ship_order=ship_order)


def arm_masks(match_keys: Sequence[str], spec: ArmSpec) -> dict[str, np.ndarray]:
    keys = np.asarray([str(k) for k in match_keys])
    return {name: np.isin(keys, sorted(members)) for name, members in spec.arms.items()}
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/hf_jobs/test_arms.py -v`
Expected: PASS (5 tests).

---

### Task 2: Persist arm masks in `StudyInputs` (byte-identical fan-out)

The fan-out study workers reconstruct candidate masks from `load_study_inputs`. Today those masks are
rebuilt from `(providers, is_public)`. Persist the arm masks themselves so workers build identical masks
with no provider logic.

**Files:**
- Modify: `scripts/_study_shared.py` (`StudyInputs` ~:29-36, `save_study_inputs` ~:54-65,
  `load_study_inputs` ~:102-112)
- Test: `tests/scripts/test_study_shared_arms.py`

**Interfaces:**
- Consumes: `scripts.hf_jobs.arms` (Task 1) is NOT imported here (keep `_study_shared` dependency-free of
  the harness); arm masks arrive as a plain `dict[str, np.ndarray]`.
- Produces: `StudyInputs.arms: dict[str, np.ndarray]`; `save_study_inputs(..., arms=...)` writes
  `arms/<name>.npy`; `load_study_inputs(...).arms` reads them back. `is_public` stays (it is
  `arms["public"]`, kept for `_public_folds`).

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_study_shared_arms.py
import numpy as np
import pandas as pd
from scripts._study_shared import load_study_inputs, save_study_inputs


def test_arms_round_trip(tmp_path):
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    y = np.array([0, 1, 0])
    groups = np.array(["g1", "g2", "g3"])
    providers = np.array(["sportec", "skillcorner", "skillcorner"])
    match_ids = np.array(["DFL-MAT-J03WMX", "1874553", "9999"])
    is_public = np.array([True, True, False])
    arms = {"public": is_public, "sc_extended": np.array([True, True, True])}
    save_study_inputs(tmp_path, X, y, groups, providers, match_ids, is_public,
                      config={"run_paired": True}, arms=arms)
    inp = load_study_inputs(tmp_path)
    assert inp.arms["public"].tolist() == [True, True, False]
    assert inp.arms["sc_extended"].tolist() == [True, True, True]
    assert inp.is_public.tolist() == [True, True, False]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/test_study_shared_arms.py -v`
Expected: FAIL (`save_study_inputs() got an unexpected keyword argument 'arms'`).

- [ ] **Step 3: Implement the change**

In `scripts/_study_shared.py`:
- Add `arms: dict[str, np.ndarray]` to the `StudyInputs` dataclass.
- `save_study_inputs(..., *, config, arms)`: after writing `is_public.npy`, write each arm:
  ```python
  arms_dir = root / "arms"
  arms_dir.mkdir(exist_ok=True)
  for name, mask in arms.items():
      np.save(arms_dir / f"{name}.npy", np.asarray(mask, dtype=bool))
  ```
- `load_study_inputs`: read them back:
  ```python
  arms_dir = root / "arms"
  arms = {p.stem: np.load(p) for p in sorted(arms_dir.glob("*.npy"))} if arms_dir.exists() else {}
  ```
  and pass `arms=arms` into `StudyInputs(...)`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/test_study_shared_arms.py -v`
Expected: PASS.

---

### Task 2.5: Relocate neutral provenance helpers out of `_corpus.py` (HVB-PLAN-01)

**Why (reviewer B):** the Task-6 closure gate flags any module *reachable* from a trainer whose source
*defines* a banned symbol. `_corpus.py` defines `PUBLIC_CORPUS`/`is_public_row`/`assert_public_corpus`/
`artifact_label`, so as long as a trainer imports **anything** from `_corpus` (even the neutral
`corpus_identity`/`reproducibility`), `_corpus.py` is in the closure and the gate fails. The trainers must
import **nothing** from `_corpus`. Relocate the neutral helpers the `--data-dir` trainer still needs.

**Verified classification (AST, `_corpus.py`):**
- NEUTRAL (no `PUBLIC_CORPUS`/`is_public_row`/`assert_public_corpus`/`match_visibility`): `corpus_identity`,
  `reproducibility` (references the *word* visibility as a param only, not the banned symbols),
  `check_shipped_variant`, `check_expected_variant`, `match_id_pairs`.
- VISIBILITY (stay in `_corpus.py`, launcher/off-HF only): `PUBLIC_CORPUS`, `is_public_row`,
  `assert_public_corpus`, `artifact_label`, `requested_is_all_public`, `bundled_public_arm_pairs`.

**Files:**
- Create: `scripts/_corpus_identity.py` (holds `corpus_identity`, `reproducibility`,
  `check_shipped_variant`, `check_expected_variant`, `match_id_pairs` — the neutral helpers).
- Modify: `scripts/_corpus.py` (remove those five defs; retain the visibility taxonomy only). If any stays
  co-dependent, re-export from `_corpus` via `from scripts._corpus_identity import ...` is NOT allowed (it
  would re-pull the symbols into `_corpus`'s namespace but not its closure risk — acceptable for the
  LAUNCHER, but the trainers must import from `_corpus_identity` directly).
- Modify: every importer of the moved helpers — find with the grep below; repoint to `_corpus_identity`.
- Test: `tests/scripts/test_corpus_identity_split.py`

**Interfaces:**
- Produces: `scripts/_corpus_identity.py` with `corpus_identity(providers, match_ids, *, all_public)`,
  `reproducibility(shipped, providers, *, training_commit)`, `check_shipped_variant(expected, shipped)`,
  `check_expected_variant(...)`, `match_id_pairs(...)` — signatures unchanged from `_corpus`.

- [ ] **Step 1: Enumerate every importer of the moved helpers**

Run: `grep -rnE "from _corpus import|_corpus\.(corpus_identity|reproducibility|check_shipped_variant|check_expected_variant|match_id_pairs)" scripts/ silly_kicks/ tests/`
Record the list; each non-trainer importer repoints to `_corpus_identity` too.

- [ ] **Step 2: Write the failing test**

```python
# tests/scripts/test_corpus_identity_split.py
import ast


def _import_names(path):
    t = ast.parse(open(path, encoding="utf-8").read())
    return {a.name for n in ast.walk(t) if isinstance(n, ast.ImportFrom) for a in n.names}


def test_corpus_identity_has_no_visibility_symbols():
    src = open("scripts/_corpus_identity.py", encoding="utf-8").read()
    for banned in ("PUBLIC_CORPUS", "is_public_row", "assert_public_corpus", "artifact_label",
                   "match_visibility"):
        assert banned not in src, banned


def test_corpus_identity_exports_the_neutral_helpers():
    import scripts._corpus_identity as ci
    for fn in ("corpus_identity", "reproducibility", "check_shipped_variant"):
        assert hasattr(ci, fn)
```

- [ ] **Step 3: Run → FAIL** (`scripts/_corpus_identity.py` missing).

- [ ] **Step 4: Implement** — cut the five neutral defs from `_corpus.py` into `scripts/_corpus_identity.py`
      (verbatim, with their imports); repoint every importer found in Step 1.

- [ ] **Step 5: Run → PASS** + the existing `_corpus` consumers still green:

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/test_corpus_identity_split.py tests/test_metric_contracts.py -k "corpus or identity or reproducib" -v`
Expected: PASS (and no import error anywhere — a repo-wide `pytest -q` collection must still succeed).

---

### Task 3: `xshot_occurrence` visibility-blind

Replace the manifest/visibility derivation and the provider-name candidate logic with the `--arms`
parameter; delete the `--providers` pining path; remove every `_corpus`/`_loader_pining` visibility import.

**Files:**
- Modify: `scripts/train_xshot_occurrence.py`
  - `main` arg parser ~:692-700, :800-801 (add `--arms`; drop `--providers`/`--match-ids-json`/
    `--max-per-provider`/`--expect-variant` pining args)
  - the visibility block ~:894-905 (`from _corpus import ...`, `from _loader_pining import ...`,
    `vis = ...`, `assert_public_corpus`, `is_public_row`)
  - `_build_candidate_masks` :473-482
  - the ship path `artifact_label` :617, `corpus_identity` :676, `reproducibility` :678
  - `_corpus_fingerprint` ~:54-64 (drop the `match_visibility` branch), `_iter_matches_from_dir` :70-74
  - the `--providers` extraction branch :847-857
- Test: `tests/scripts/test_xshot_visibility_blind.py`

**Interfaces:**
- Consumes: `scripts.hf_jobs.arms.{load_arms, arm_masks, match_key}` (Task 1); `StudyInputs.arms` (Task 2).
- Produces: `_build_candidate_masks(match_keys, armspec) -> dict[str, np.ndarray]` (new signature — masks
  from the arm spec, NOT `(providers, is_public)`); `main` requires `--arms` on `--data-dir`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/scripts/test_xshot_visibility_blind.py
import ast
import numpy as np
import pytest
from scripts.hf_jobs.arms import ArmSpec
from scripts.train_xshot_occurrence import _build_candidate_masks


def test_candidate_masks_from_arms_not_providers():
    keys = ["sportec__DFL-MAT-J03WMX", "skillcorner__1874553", "skillcorner__9999"]
    spec = ArmSpec(arms={"public": frozenset(keys[:2]), "sc_extended": frozenset(keys)},
                   ship_order=("public", "sc_extended"))
    masks = _build_candidate_masks(keys, spec)
    assert masks["public"].tolist() == [True, True, False]
    assert masks["sc_extended"].tolist() == [True, True, True]


def test_trainer_source_has_no_visibility_imports():
    src = open("scripts/train_xshot_occurrence.py", encoding="utf-8").read()
    tree = ast.parse(src)
    banned = {"match_visibility", "is_public_row", "assert_public_corpus", "artifact_label"}
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not (names & banned), f"visibility symbols imported: {names & banned}"
    assert "PUBLIC_CORPUS" not in src
    assert "match_visibility" not in src
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/test_xshot_visibility_blind.py -v`
Expected: FAIL (`_build_candidate_masks` old signature; visibility symbols still imported).

- [ ] **Step 3: Implement**

1. Arg parser: add `ap.add_argument("--arms")`. Delete `--providers`, `--match-ids-json`,
   `--max-per-provider`, `--expect-variant` and the `src` mutually-exclusive `--providers` option. The
   source is now `--data-dir` only; keep the existing `--shard-index`/`--prep-only`/`--study`/`--assemble`
   phase flags.
2. `_corpus_fingerprint`: delete the `match_visibility`/`select_match_ids` import + the `if args.providers`
   branch; fingerprint the `--data-dir` contents only.
3. Extraction: delete the `if args.providers: ... pining_source(...)` branch (:847-857); keep
   `source, load = _iter_matches_from_dir(Path(args.data_dir)), None`.
4. Replace the visibility block (:894-905) with:
   ```python
   from scripts.hf_jobs.arms import arm_masks, load_arms, match_key

   if not args.arms:
       sys.exit("ERROR: --arms is required (ADR-123 Amendment B: visibility enters as a job parameter).")
   armspec = load_arms(args.arms)
   match_keys = np.array([match_key(str(p), str(m)) for p, m in zip(providers, match_ids, strict=True)])
   masks = arm_masks(match_keys, armspec)
   is_public = masks.get("public", np.zeros(len(match_keys), bool))
   ```
   Delete `from _corpus import assert_public_corpus, is_public_row`, `from _loader_pining import
   match_visibility`, the `vis = ...` line, and the `assert_public_corpus(...)` call.
5. `_build_candidate_masks(match_keys, armspec) -> dict`: return `arm_masks(match_keys, armspec)` (drop the
   `providers == "skillcorner"` logic). Update its three callers (`main`, `enumerate_studies`,
   `run_one_study`, `assemble_studies`) to pass `(inp-or-local match_keys, armspec)`. Persist `arms=masks`
   + `match_keys` via `save_study_inputs` (Task 2) so the fan-out path reconstructs them; `enumerate_studies`/
   `run_one_study`/`assemble_studies` read `inp.arms` instead of rebuilding from providers.
6. Ship path: the shipped arm is the first `ship_order` arm that passes gates (existing preference logic,
   now walking `armspec.ship_order`); its label is the arm NAME:
   ```python
   shipped = <arm name chosen from armspec.ship_order>   # replaces artifact_label(providers=, all_public=)
   ```
   `corpus_identity(providers.tolist(), match_ids.tolist(), all_public=bool(is_public.all()))` keeps
   `providers`/`match_ids` as PROVENANCE data (they are data, not policy) but `all_public` comes from
   `is_public` (the `public`-arm mask), not from provider names. `reproducibility(shipped,
   providers_of_ship_mask, ...)` keeps provider names as provenance only.
   **Imports (HVB-PLAN-01):** `corpus_identity`, `reproducibility`, `check_shipped_variant` are imported
   from `scripts._corpus_identity` (Task 2.5), NOT `_corpus`. After this task the trainer imports
   **nothing** from `_corpus` and nothing from `_loader_pining` — the Task-6 closure gate then passes.
8. `main` for the `--data-dir`-only trainer: when `run_paired` (more than one arm present) run the paired
   comparison over `armspec.arms`; else single-candidate (the all-public case) as today.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/test_xshot_visibility_blind.py -v`
Expected: PASS.

- [ ] **Step 5: Run the trainer's existing unit/smoke tests**

Run: `.venv\Scripts\python.exe -m pytest tests/ -k "xshot" -v --no-header`
Expected: PASS (fix any that referenced `--providers`/old `_build_candidate_masks`).

---

### Task 4: `xcross_attempt` visibility-blind (mirror of Task 3)

xcross mirrors xshot; apply the identical refactor at the mirror sites, and keep the agnostic
`--probe-data-dir` path (ADR-123) untouched.

**Files:**
- Modify: `scripts/train_xcross_attempt.py`
  - `match_visibility`/`select_match_ids` import :53, `vis = match_visibility(...)` :58
  - `from _corpus import artifact_label, check_shipped_variant, corpus_identity, reproducibility` :574
  - `artifact_label` :677, `corpus_identity` :737
  - `_build_candidate_masks` :483-489, `_iter_matches_from_dir` :62
  - the `--providers` arg :825 and its paths :976-977, :991-992, :1030-1039, :1069 (B flagged the two
    code paths — the main extraction AND the `--expect-variant`/probe-listing branch :1181 region)
- Test: `tests/scripts/test_xcross_visibility_blind.py`

**Interfaces:**
- Consumes/Produces: identical to Task 3 (`_build_candidate_masks(match_keys, armspec)`; `--arms` required
  on `--data-dir`). The xcross `--probe-data-dir` stays as the ADR-123 held-out TF-19 probe mechanism.

- [ ] **Step 1: Write the failing tests** — copy `tests/scripts/test_xshot_visibility_blind.py`
      substituting `train_xcross_attempt` (repeat the code; do not reference Task 3's test):

```python
# tests/scripts/test_xcross_visibility_blind.py
import ast
from scripts.hf_jobs.arms import ArmSpec
from scripts.train_xcross_attempt import _build_candidate_masks


def test_candidate_masks_from_arms_not_providers():
    keys = ["sportec__DFL-MAT-J03WMX", "skillcorner__1874553", "skillcorner__9999"]
    spec = ArmSpec(arms={"public": frozenset(keys[:2]), "sc_extended": frozenset(keys)},
                   ship_order=("public", "sc_extended"))
    masks = _build_candidate_masks(keys, spec)
    assert masks["public"].tolist() == [True, True, False]
    assert masks["sc_extended"].tolist() == [True, True, True]


def test_xcross_source_has_no_visibility_imports():
    src = open("scripts/train_xcross_attempt.py", encoding="utf-8").read()
    names = {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not (names & {"match_visibility", "is_public_row", "assert_public_corpus", "artifact_label"})
    assert "PUBLIC_CORPUS" not in src and "match_visibility" not in src
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/test_xcross_visibility_blind.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** — apply the Task 3 Step-3 changes (items 1-8, incl. importing
      `corpus_identity`/`reproducibility`/`check_shipped_variant` from `scripts._corpus_identity`, NOT
      `_corpus`) at the xcross mirror sites. After this task xcross imports **nothing** from `_corpus`/
      `_loader_pining`. Keep `--probe-data-dir` and its `_write_probe_sample` path exactly. The probe
      extraction derives its own cohort from the staged probe dir, independent of `--arms`.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/test_xcross_visibility_blind.py -v`
Expected: PASS.

- [ ] **Step 5: Run the trainer's existing tests**

Run: `.venv\Scripts\python.exe -m pytest tests/ -k "xcross" -v --no-header`
Expected: PASS.

---

### Task 5: Launcher computes arms off-HF + threads `--arms` (HVB-SPEC-01)

The only place the public/owner taxonomy lives for the HF path. Maps `sportec`→`idsse` INTERNALLY for the
`PUBLIC_CORPUS` lookup and EMITS the dataset's own keys (B's HVB-SPEC-01 — otherwise `np.isin` misses the
DFL rows and the public arm is empty: the fixed bug, relocated).

**Files:**
- Create: `scripts/hf_jobs/arm_policy.py`
- Modify: `scripts/hf_jobs/launch.py` (`_corpus_steps`/`build_phase_command` — thread `--arms`;
  mirror the `probe_repo`→`--probe-data-dir` pattern at :137-141), `scripts/hf_jobs/registry.py`
  (`TrainerSpec` + `visibility_gated: bool = False`; set `True` on `xshot_occurrence`/`xcross_attempt`)
- Test: `tests/scripts/hf_jobs/test_arm_policy.py`, extend `tests/scripts/hf_jobs/test_launch.py`

**Interfaces:**
- Consumes: `scripts.hf_jobs.arms.match_key` (Task 1), `scripts._corpus.PUBLIC_CORPUS`.
- Produces: `compute_arms(corpus_keys: Sequence[str]) -> dict` — the `--arms` JSON payload (arms +
  ship_order) built from `PUBLIC_CORPUS`, keyed by DATASET keys; `_CONVERTER_TO_IDENTITY = {"sportec":
  "idsse"}` (the launcher-only provider-identity map).

- [ ] **Step 1: Write the failing tests**

```python
# tests/scripts/hf_jobs/test_arm_policy.py
from scripts.hf_jobs.arm_policy import compute_arms


def test_public_arm_uses_dataset_keys_not_registry_identity():
    # staged corpus keys the Bundesliga match as sportec__ (the converter), PUBLIC_CORPUS keys it idsse__.
    corpus_keys = ["sportec__DFL-MAT-J03WMX", "skillcorner__1874553", "skillcorner__9999"]
    payload = compute_arms(corpus_keys)
    public = set(payload["arms"]["public"])
    # the DFL match is public (idsse in PUBLIC_CORPUS) AND emitted under its DATASET key (sportec__):
    assert "sportec__DFL-MAT-J03WMX" in public
    assert "idsse__DFL-MAT-J03WMX" not in public            # never the registry identity
    assert "skillcorner__1874553" in public                 # registered public skillcorner
    assert "skillcorner__9999" not in public                # unregistered -> owner/private
    assert set().union(*payload["arms"].values()) <= set(corpus_keys)   # arm keys subset of corpus keys


def test_all_public_corpus_single_public_arm():
    corpus_keys = ["sportec__DFL-MAT-J03WMX", "skillcorner__1874553"]  # both registered public
    payload = compute_arms(corpus_keys)
    assert set(payload["arms"]["public"]) == set(corpus_keys)
```

```python
# add to tests/scripts/hf_jobs/test_launch.py
def test_visibility_gated_threads_arms_from_dataset_keys(tmp_path):
    import scripts.hf_jobs.registry as reg
    from scripts.hf_jobs.launch import build_inner_command
    spec = reg.REGISTRY["xshot_occurrence"]
    inner = build_inner_command(spec, spec.builds[0], tag="r", run_key="r1", corpus_repo="ns/xshot17")
    assert "--arms /tmp/arms.json" in inner
    assert reg.REGISTRY["xshot_occurrence"].visibility_gated is True
    assert reg.REGISTRY["ghost_gk"].visibility_gated is False


def test_arm_keys_subset_of_corpus_keys_guard():
    # HVB-SPEC-01: every arm key must be a staged-corpus key (else np.isin misses -> empty public arm).
    from scripts.hf_jobs.arm_policy import compute_arms
    corpus = ["skillcorner__1874553", "sportec__DFL-MAT-J03WMX"]
    assert set().union(*compute_arms(corpus)["arms"].values()) <= set(corpus)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/hf_jobs/test_arm_policy.py tests/scripts/hf_jobs/test_launch.py -v`
Expected: FAIL (`scripts.hf_jobs.arm_policy` missing; `visibility_gated` missing; `--arms` not threaded).

- [ ] **Step 3: Implement**

`scripts/hf_jobs/arm_policy.py`:
```python
"""Launcher-side (OFF-HF) public/owner arm policy. The ONLY place the HF path touches the taxonomy.

Reads the in-repo redistributability registry (_corpus.PUBLIC_CORPUS) and emits a neutral --arms payload
keyed by the DATASET's own match keys. The converter->identity map (sportec->idsse) is applied INTERNALLY
for the registry lookup only; emitted keys are always the dataset keys (HVB-SPEC-01)."""

from __future__ import annotations

from collections.abc import Sequence

from scripts._corpus import PUBLIC_CORPUS

_CONVERTER_TO_IDENTITY = {"sportec": "idsse"}  # DFL/Bundesliga: source_provider is the converter name


def _is_public(provider: str, match_id: str) -> bool:
    identity = _CONVERTER_TO_IDENTITY.get(provider, provider)
    return match_id in PUBLIC_CORPUS.get(identity, frozenset())


def compute_arms(corpus_keys: Sequence[str]) -> dict:
    public, extended, full = [], [], []
    for key in corpus_keys:
        provider, _, match_id = key.partition("__")
        full.append(key)
        if _is_public(provider, match_id):
            public.append(key)
            extended.append(key)
        elif provider == "skillcorner":            # owner-tier SkillCorner -> sc_extended only
            extended.append(key)
    arms = {"public": public}
    if set(extended) != set(public):
        arms["sc_extended"] = extended
    if set(full) != set(extended):
        arms["full"] = full
    arms = {k: v for k, v in arms.items() if v}
    return {"arms": arms, "ship_order": [n for n in ("public", "sc_extended", "full") if n in arms]}
```

`registry.py`: add `visibility_gated: bool = False` to `TrainerSpec`; set `True` on `xshot_occurrence`
and `xcross_attempt`.

`launch.py` — the computation is LOCAL (off-HF), the result is a RECEIVED parameter (owner directive:
"if a result must carry public/private, it comes from a parameter the HF received with the job"; the job
must never compute the policy):
- `launch.py` runs on the local launcher. A helper `_corpus_match_keys(corpus_repo) -> list[str]` lists
  the staged dataset's shard filenames via `huggingface_hub.HfApi().list_repo_files(corpus_repo,
  repo_type="dataset")`, deriving each dataset key `<source_provider>__<match_id>` from the
  `shards/<gen>/<provider>__<id>.parquet` names.
- For a `visibility_gated` spec, `build_phase_command` calls `compute_arms(_corpus_match_keys(corpus_repo))`
  LOCALLY and embeds the resulting JSON into the inner command as a heredoc written to `/tmp/arms.json`:
  ```python
  arms_json = json.dumps(compute_arms(corpus_keys))
  prep.append(f"cat > /tmp/arms.json <<'SK_ARMS_EOF'\n{arms_json}\nSK_ARMS_EOF")
  ```
  then append `--arms /tmp/arms.json` to the trainer command (mirrors the `--probe-data-dir` threading).
- `arm_policy` / `_corpus.PUBLIC_CORPUS` are therefore imported ONLY on the local launcher, NEVER executed
  in-job: the job receives `/tmp/arms.json` as pre-written data (a parameter), exactly as the owner
  directed. Task 6's closure gate confirms neither `arm_policy` nor `_corpus` nor `_loader_pining` is in
  the TRAINER's runtime import closure (they stay present in the cloned repo but unexecuted).

> The alternative of running `compute_arms` in the inner bash (reading `PUBLIC_CORPUS` in the cloned repo)
> is REJECTED: it would have the job compute the policy, which violates the owner directive. Local
> computation + heredoc-embedded `/tmp/arms.json` is the design, not a reviewer choice.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/hf_jobs/test_arm_policy.py tests/scripts/hf_jobs/test_launch.py -v`
Expected: PASS.

---

### Task 6: Transitive I1 import gate with teeth (HFVB-SPEC-01)

A top-level-only check is vacuous (B/A). Walk the trainer's transitive import closure and fail on ANY
module that imports a banned visibility symbol; plant a leak in a fixture and assert the gate catches it.

**Files:**
- Test: `tests/scripts/hf_jobs/test_visibility_blind_gate.py`
- Create (fixture): `tests/scripts/hf_jobs/_leak_fixtures/leaky_module.py`

**Interfaces:**
- Consumes: nothing new; walks `scripts/train_xshot_occurrence.py` + `scripts/train_xcross_attempt.py`
  import graphs.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/hf_jobs/test_visibility_blind_gate.py
import ast
from pathlib import Path

BANNED = {"match_visibility", "is_public_row", "assert_public_corpus", "artifact_label", "PUBLIC_CORPUS"}
SCRIPTS = Path("scripts")


REPO = Path(".")


def _resolve(mod: str, importer: Path) -> Path | None:
    """Resolve a dotted or bare module to a real file (HFVB-PLAN-01: subpackage-aware).

    `scripts.hf_jobs.arm_policy` -> scripts/hf_jobs/arm_policy.py; a bare `from _corpus import` ->
    scripts/_corpus.py (sibling of the importer, matching the trainers' sys.path.insert(0, 'scripts'))."""
    if mod.startswith("scripts."):
        p = REPO / Path(mod.replace(".", "/") + ".py")
        return p if p.exists() else (REPO / Path(mod.replace(".", "/")) / "__init__.py" if (
            REPO / Path(mod.replace(".", "/")) / "__init__.py").exists() else None)
    # bare name: a scripts/ sibling (how the trainers import _corpus / _loader_pining)
    p = SCRIPTS / f"{mod.split('.')[-1]}.py"
    return p if p.exists() else None


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)                       # keep the FULL dotted path (no .split last)
        elif isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
    return mods


def _closure(entry: str) -> set[Path]:
    seen, stack = set(), [SCRIPTS / f"{entry}.py"]
    while stack:
        p = stack.pop()
        if p in seen or not p.exists():
            continue
        seen.add(p)
        for mod in _imported_modules(p):
            nxt = _resolve(mod, p)
            if nxt is not None:
                stack.append(nxt)
    return seen


def _banned_hits(path: Path) -> set[str]:
    src = path.read_text(encoding="utf-8")
    names = {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) for a in n.names}
    return (names & BANNED) | {b for b in BANNED if b in src}


def test_xshot_closure_is_visibility_blind():
    hits = {p.name: _banned_hits(p) for p in _closure("train_xshot_occurrence") if _banned_hits(p)}
    assert not hits, f"visibility concept reachable from the HF job: {hits}"


def test_xcross_closure_is_visibility_blind():
    hits = {p.name: _banned_hits(p) for p in _closure("train_xcross_attempt") if _banned_hits(p)}
    assert not hits, f"visibility concept reachable from the HF job: {hits}"


def test_gate_traverses_a_subpackage_hop(tmp_path, monkeypatch):
    # anti-rot (HFVB-PLAN-01): prove the TRAVERSAL, not just per-file detection. entry.py -> imports
    # scripts.hf_jobs.leaky (a SUBPACKAGE hop) -> leaky.py imports match_visibility. _closure MUST reach
    # leaky.py (so a future indirect leak via any scripts/hf_jobs/ module is caught).
    import tests.scripts.hf_jobs.test_visibility_blind_gate as gate
    root = tmp_path
    (root / "scripts" / "hf_jobs").mkdir(parents=True)
    (root / "scripts" / "entry.py").write_text("from scripts.hf_jobs.leaky import f\n")
    (root / "scripts" / "hf_jobs" / "leaky.py").write_text(
        "from _loader_pining import match_visibility\ndef f():\n    return match_visibility\n")
    monkeypatch.setattr(gate, "REPO", root)
    monkeypatch.setattr(gate, "SCRIPTS", root / "scripts")
    closure = _closure("entry")
    leaky = root / "scripts" / "hf_jobs" / "leaky.py"
    assert leaky in closure, "closure did not traverse the subpackage hop"
    assert any(_banned_hits(p) for p in closure), "gate missed the indirect leak behind the hop"
```

- [ ] **Step 2: Run**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/hf_jobs/test_visibility_blind_gate.py -v`
Expected: the two closure tests FAIL before Tasks 2.5/3/4 land (or PASS after), the subpackage-hop
traversal test PASSES (it uses a tmp tree, independent of the real code).

- [ ] **Step 3: Make the closure tests pass** — satisfied by Tasks 2.5 + 3 + 4: Task 2.5 removes `_corpus`
      from the trainer closure (the SHOULD-FIX), Tasks 3/4 strip the visibility imports and `--providers`.
      `arm_policy` imports `PUBLIC_CORPUS` but is launcher-only (Task 5: `compute_arms` runs on the local
      launcher, `/tmp/arms.json` is embedded data) — confirm the closure walk does NOT reach `arm_policy`
      from either trainer (it is not imported by the trainer; the gate now resolves subpackages, so this is
      a real check, not a blind spot).

- [ ] **Step 4: Run to verify all pass**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/hf_jobs/test_visibility_blind_gate.py -v`
Expected: PASS (closures blind, planted leak caught).

---

### Task 7: Fail-loud + faithfulness + smoke-path note (I4, HFVB-SPEC-02)

**Files:**
- Test: `tests/scripts/test_xshot_arms_behavior.py`
- Modify (doc): the spec's §4.4 smoke note is already present; add the same one-liner to
  `docs/howto/hf-jobs-training.md` (the `--data-dir` smoke example now passes `--arms`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/scripts/test_xshot_arms_behavior.py
import subprocess, sys


def test_data_dir_without_arms_fails_loud(tmp_path):
    (tmp_path / "g1").mkdir()
    r = subprocess.run([sys.executable, "-m", "scripts.train_xshot_occurrence",
                        "--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "--arms is required" in (r.stdout + r.stderr)
```

Faithfulness (single-public ship) is covered by the Task 3 `_build_candidate_masks` test plus the Task 5
`test_all_public_corpus_single_public_arm`; together they prove: all-public corpus → one `public` arm →
`public` shipped. A full end-to-end faithfulness check (PR-AUC ≈ committed) is the DGX e2e (Task 8) — it
needs real data and is not a unit test.

- [ ] **Step 2: Run to verify it fails** → FAIL (today it silently runs all-private).
- [ ] **Step 3:** implemented by Task 3 item 4 (the `sys.exit("ERROR: --arms is required ...")`).
- [ ] **Step 4: Run to verify it passes** → PASS.
- [ ] **Step 5:** update `docs/howto/hf-jobs-training.md` smoke example to pass `--arms`.

---

### Task 8: DGX e2e extended — shipped-candidate == committed

Close the GATE-1 gap (exit-0 only). The e2e asserts the fresh shipped candidate NAME equals the committed
shipped candidate (`public`) for xshot/xcross, and records CV-parity (fresh vs committed
`candidates.public.metrics.pr_auc` within the per-trainer δ).

**Files:**
- Modify: `scripts/hf_jobs/dgx_e2e.py` (add a `shipped_variant` assertion to `run_local_e2e` for
  visibility-gated specs)
- Test: `tests/scripts/hf_jobs/test_dgx_e2e.py` (unit-level: the assertion fires on a mismatch)

- [ ] **Step 1: Write the failing test** (a stub run-record with `shipped_variant="sc_extended"` must be
      flagged blocked when the committed is `public`):

```python
def test_e2e_flags_wrong_shipped_candidate():
    from scripts.hf_jobs.dgx_e2e import check_shipped_variant
    assert check_shipped_variant(fresh="public", committed="public") is None
    err = check_shipped_variant(fresh="sc_extended", committed="public")
    assert err and "sc_extended" in err and "public" in err
```

- [ ] **Step 2: Run → FAIL** (`check_shipped_variant` missing).
- [ ] **Step 3: Implement** `check_shipped_variant(*, fresh, committed) -> str | None` and call it in
      `run_local_e2e` for `visibility_gated` specs; a mismatch sets `summary["blocked"] = True`.
- [ ] **Step 4: Run → PASS.**

---

### Task 9: Docs — ADR-123 Amendment B + CHANGELOG + maintainer DGX note

**Files:**
- Modify: `docs/superpowers/adrs/ADR-123-hf-jobs-parallel-harness.md` (add "Amendment B: visibility-blind
  in-job code; policy via `--arms`; `--providers` removed from the trainers; maintainer DGX pining
  workflow uses the launcher + `--arms`")
- Modify: `docs/howto/hf-jobs-training.md` (the `--arms` contract; the maintainer DGX re-fit now stages +
  computes arms + calls the agnostic trainer, not `--providers`)
- Modify: `CHANGELOG.md` (one `PR-Snnn` entry; no version claim until the release)
- Modify: `AGENTS.md` — the xshot/xcross bullets note the arm-parameter contract if the terse rule bites
  (budget-checked by `tests/test_agents_md_budget.py`).

- [ ] **Step 1:** write the ADR-123 Amendment B section (the invariants I1-I4, the `--arms` schema, the
      `--providers`-removal + maintainer-DGX consequence, the root cause, the GATE-1-gap closure).
- [ ] **Step 2:** update the howto + CHANGELOG + AGENTS.md.
- [ ] **Step 3: Run the doc-wiring gates**

Run: `.venv\Scripts\python.exe -m pytest tests/test_howto_links_wired.py tests/test_agents_md_budget.py tests/test_public_api_examples.py -v`
Expected: PASS.

---

## Final verification (before the single owner-gated commit)

- [ ] **HVB-PLAN-02 — `--providers` live-caller enumeration (spec R2/§6):** before deleting the trainers'
      `--providers` path, confirm no LIVE caller depends on it:
      `grep -rnE "train_(xshot_occurrence|xcross_attempt).*--providers" .github/ scripts/ Makefile* *.ps1 *.sh 2>/dev/null`
      — expect only historical `docs/`/`plans/` references (no CI/script caller). The maintainer DGX
      pining re-fit adopts the launcher + `--arms` (documented in Task 9). Record the grep result.
- [ ] `.venv\Scripts\python.exe -m pytest tests/ -m "not e2e" -q` — full suite green (CI authoritative;
      the ghost determinism `@slow` tests skip under sklearn<1.9 locally).
- [ ] `python -m ruff check silly_kicks/ tests/ scripts/` + `python -m ruff format --check silly_kicks/ tests/ scripts/` clean.
- [ ] `pyright` on every changed file — 0 errors (scoped; local phantom errors on unrelated files are CI-authoritative, never edited).
- [ ] The I1 closure gate (Task 6) green for BOTH trainers; the planted-leak meta-assertion green.
- [ ] DGX e2e (Task 8): run xshot + xcross against the staged corpus; `shipped_variant == "public"` and
      CV-parity `public` PR-AUC ≈ committed (0.34 default / 0.30 position_only) within δ. If not, STOP and
      escalate (the staged corpus differs beyond the label — spec §10 R3).
- [ ] FREEZE the tree → independent impl review (author does NOT run own review) → owner go.
- [ ] Re-fit xshot + xcross on HF (agnostic, `--arms` threaded) → CV-parity confirm → retrieve weights.
- [ ] ONE owner-gated commit, bundled with the weights commit; no amend of `6f6508a`/`ac875c4`; merge #284
      non-squash per ADR-123 provenance.

## Self-review notes (author)

- **Spec coverage:** I1 (Task 6 gate + Tasks 3/4 strip), I2 (Tasks 1/5 `--arms`), I3 (no dataset change —
  Task 5 payload is a parameter), I4 (Task 3 item 4 + Task 7), HVB-SPEC-01 (Task 5 dataset-key emit +
  subset test), HFVB-SPEC-01 (Task 6 transitive + planted leak), HFVB-SPEC-02 (Task 7 smoke note), the
  shipped-candidate e2e (Task 8), `--providers` removal + maintainer consequence (Tasks 3/4/9), provenance
  (Final verification). No gap.
- **Policy computed OFF-HF (Task 5):** `compute_arms`/`PUBLIC_CORPUS` run only on the local launcher;
  `/tmp/arms.json` is embedded as a received parameter (heredoc) — the job never computes the policy, per
  the owner directive. `arm_policy`/`_corpus`/`_loader_pining` stay out of the trainer's runtime closure
  (Task 6 gate).
- **Byte-identity:** arms persisted in `StudyInputs` (Task 2) so the fan-out study workers rebuild
  identical masks — the ADR-123 determinism contract holds for the paired comparison.
- **Review r1 folded:** HVB-PLAN-01/B (SHOULD-FIX) → new Task 2.5 relocates the neutral helpers out of
  `_corpus.py` so the trainers import nothing from it (the closure gate then passes; Task 3 item 7 /
  Task 4 repoint to `_corpus_identity`; Task 6 Step 3 corrected — no longer claims "Tasks 3-4" alone).
  HFVB-PLAN-01/A → Task 6 `_resolve`/`_imported_modules` now resolve DOTTED subpackage paths + a
  subpackage-hop traversal meta-assertion. HVB-PLAN-02/B → `--providers` live-caller grep in Final
  verification. HFVB-SPEC-01/02 + HVB-SPEC-01 folded as before.
