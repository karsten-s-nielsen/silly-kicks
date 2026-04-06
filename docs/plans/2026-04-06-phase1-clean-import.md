# Phase 1: Clean Import — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a working, testable `silly-kicks` package that is functionally identical to socceraction v1.5.3 but with modern build tooling, renamed imports, and proper attribution.

**Architecture:** Detached fork — clone upstream source, strip git history, rename `socceraction` to `silly_kicks` throughout, replace Poetry build system with hatchling, set up Ruff/Pyright/pytest CI. Zero behavioral changes; the existing test suite is the proof.

**Tech Stack:** Python 3.10+, hatchling (build), Ruff (lint), Pyright (types), pytest (tests), GitHub Actions (CI). Dependencies: pandas, numpy, scikit-learn, lxml, pandera.

**Working directory:** `D:\Development\karstenskyt__silly-kicks\`

**Spec:** `docs/specs/2026-04-06-silly-kicks-design.md`

---

## File Structure (Phase 1 end state)

```
karstenskyt__silly-kicks/
├── silly_kicks/                 # Renamed from socceraction/
│   ├── __init__.py
│   ├── xthreat.py
│   ├── atomic/
│   │   ├── __init__.py
│   │   ├── spadl/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── config.py
│   │   │   ├── schema.py
│   │   │   └── utils.py
│   │   └── vaep/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── features.py
│   │       ├── formula.py
│   │       └── labels.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── schema.py
│   │   ├── opta/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py
│   │   │   ├── schema.py
│   │   │   └── parsers/
│   │   │       ├── __init__.py
│   │   │       ├── base.py
│   │   │       ├── f1_json.py
│   │   │       ├── f24_json.py
│   │   │       ├── f24_xml.py
│   │   │       ├── f7_xml.py
│   │   │       ├── f9_json.py
│   │   │       ├── ma1_json.py
│   │   │       ├── ma3_json.py
│   │   │       └── whoscored.py
│   │   ├── statsbomb/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py
│   │   │   └── schema.py
│   │   └── wyscout/
│   │       ├── __init__.py
│   │       ├── loader.py
│   │       └── schema.py
│   ├── spadl/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── config.py
│   │   ├── kloppy.py
│   │   ├── opta.py
│   │   ├── schema.py
│   │   ├── statsbomb.py
│   │   ├── utils.py
│   │   └── wyscout.py
│   └── vaep/
│       ├── __init__.py
│       ├── base.py
│       ├── features.py
│       ├── formula.py
│       └── labels.py
├── tests/                       # Upstream tests, imports renamed
│   ├── conftest.py
│   ├── test_xthreat.py
│   ├── atomic/
│   ├── data/
│   ├── spadl/
│   ├── vaep/
│   └── datasets/                # Committed test fixtures (JSON, XML)
├── docs/
│   ├── specs/                   # Design spec
│   └── plans/                   # This plan
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml               # Hatchling build, all config
├── LICENSE                      # MIT, original copyright preserved
├── README.md                    # Attribution, usage, BibTeX
└── .gitignore
```

---

## Task 1: Clone Upstream Source Into Repo

**Files:**
- Create: all files under `silly_kicks/`, `tests/`, `LICENSE`
- Source: `https://github.com/ML-KULeuven/socceraction` (tag v1.5.3)

- [ ] **Step 1: Clone upstream to a temp directory**

```bash
cd /d/Development
git clone --depth 1 --branch v1.5.3 https://github.com/ML-KULeuven/socceraction.git _socceraction_upstream
```

- [ ] **Step 2: Copy source, tests, and fixtures into silly-kicks**

```bash
cd /d/Development/karstenskyt__silly-kicks

# Copy package source (will be renamed in Task 2)
cp -r /d/Development/_socceraction_upstream/socceraction ./socceraction

# Copy tests + fixtures
cp -r /d/Development/_socceraction_upstream/tests ./tests

# Copy license (will be reformatted in Task 5)
cp /d/Development/_socceraction_upstream/LICENSE.rst ./LICENSE.rst
```

- [ ] **Step 3: Clean up temp clone**

```bash
rm -rf /d/Development/_socceraction_upstream
```

- [ ] **Step 4: Remove __pycache__ directories**

```bash
cd /d/Development/karstenskyt__silly-kicks
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

- [ ] **Step 5: Initialize git and make attribution commit**

```bash
cd /d/Development/karstenskyt__silly-kicks
git init
git add socceraction/ tests/ LICENSE.rst docs/
git commit -m "$(cat <<'EOF'
chore: import socceraction v1.5.3 source

Imported verbatim from ML-KULeuven/socceraction tag v1.5.3.
Original authors: Tom Decroos and Pieter Robberechts (KU Leuven).
Licensed under MIT. This commit preserves the upstream source
unmodified for clear attribution and diffing.

Design spec and implementation plan in docs/.
EOF
)"
```

This commit establishes provenance — anyone can diff `socceraction/` against
upstream v1.5.3 and confirm the source is identical. The `docs/` directory
contains the design spec and this implementation plan.

---

## Task 2: Rename Package (socceraction → silly_kicks)

**Files:**
- Rename: `socceraction/` → `silly_kicks/`
- Modify: all `.py` files under `silly_kicks/` and `tests/`

- [ ] **Step 1: Rename the package directory**

```bash
cd /d/Development/karstenskyt__silly-kicks
mv socceraction silly_kicks
```

- [ ] **Step 2: Rename all imports in source files**

Replace `socceraction` → `silly_kicks` in every `.py` file under `silly_kicks/`:

```bash
cd /d/Development/karstenskyt__silly-kicks
find silly_kicks -name "*.py" -exec sed -i 's/socceraction/silly_kicks/g' {} +
```

Verify no stale references remain:

```bash
grep -r "socceraction" silly_kicks/ --include="*.py"
```

Expected: zero matches.

- [ ] **Step 3: Rename all imports in test files**

```bash
cd /d/Development/karstenskyt__silly-kicks
find tests -name "*.py" -exec sed -i 's/socceraction/silly_kicks/g' {} +
```

Verify:

```bash
grep -r "socceraction" tests/ --include="*.py"
```

Expected: zero matches.

- [ ] **Step 4: Update __init__.py version string**

Edit `silly_kicks/__init__.py`. Change the docstring module name from
`socceraction` to `silly_kicks` if present. Keep `__version__ = "1.5.3"` for
now (will be updated to `0.1.0` in Task 6).

- [ ] **Step 5: Verify the rename is complete**

Search entire repo for any remaining `socceraction` references (excluding
LICENSE.rst which has the original attribution — that's intentional):

```bash
grep -r "socceraction" . --include="*.py" --include="*.toml" --include="*.yml" --include="*.md"
```

Expected: only `LICENSE.rst` and docs files should reference `socceraction`
(as attribution text, not as import paths).

- [ ] **Step 6: Commit the rename**

```bash
cd /d/Development/karstenskyt__silly-kicks
git add -A
git commit -m "$(cat <<'EOF'
refactor: rename socceraction -> silly_kicks

Mechanical find-and-replace of all package references.
Directory renamed, all internal and test imports updated.
Zero behavioral changes — purely a namespace rename.
EOF
)"
```

---

## Task 3: Create pyproject.toml (Poetry → Hatchling)

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Create pyproject.toml with hatchling build system**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "silly-kicks"
version = "0.1.0"
description = "Classify and value on-ball football actions using SPADL and VAEP"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "Karsten S. Nielsen" },
]
keywords = ["soccer", "football", "analytics", "SPADL", "VAEP", "actions"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering",
]
dependencies = [
    "pandas>=2.1.1",
    "numpy>=1.26.0",
    "scikit-learn>=1.3.1",
    "lxml>=4.9.3",
    "pandera>=0.17.2,<1.0",
]

[project.optional-dependencies]
statsbomb = ["statsbombpy>=1.11.0"]
kloppy = ["kloppy>=3.15.0"]
xgboost = ["xgboost>=2.0.0"]
hdf = ["tables>=3.8.0"]
test = [
    "pytest>=7.4.2",
    "pytest-mock>=3.11.1",
    "pytest-cov>=4.1.0",
]
dev = [
    "ruff>=0.8.0",
    "pyright>=1.1.380",
]

[project.urls]
Homepage = "https://github.com/karsten-s-nielsen/silly-kicks"
Repository = "https://github.com/karsten-s-nielsen/silly-kicks"

[tool.hatch.build.targets.wheel]
packages = ["silly_kicks"]

# --- Ruff ---
[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "S", "RUF"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]  # assert is fine in tests

[tool.ruff.lint.isort]
known-first-party = ["silly_kicks"]

# --- Pyright ---
[tool.pyright]
include = ["silly_kicks"]
pythonVersion = "3.10"
typeCheckingMode = "basic"

# --- Pytest ---
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "e2e: end-to-end tests requiring downloaded datasets",
]
filterwarnings = [
    "ignore::DeprecationWarning:pandera",
]
```

- [ ] **Step 2: Commit pyproject.toml**

```bash
cd /d/Development/karstenskyt__silly-kicks
git add pyproject.toml
git commit -m "$(cat <<'EOF'
build: add pyproject.toml with hatchling

Migrated from Poetry to hatchling build system.
Ruff (E,W,F,I,N,UP,B,S,RUF), Pyright basic mode, pytest config.
Same runtime dependencies as socceraction v1.5.3.
Python >=3.10 (raised from >=3.9).
EOF
)"
```

---

## Task 4: Create .gitignore

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg

# Virtual environments
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/

# OS
.DS_Store
Thumbs.db

# Downloaded test datasets (not committed)
tests/datasets/statsbomb/
tests/datasets/wyscout_public/
```

- [ ] **Step 2: Commit .gitignore**

```bash
cd /d/Development/karstenskyt__silly-kicks
git add .gitignore
git commit -m "chore: add .gitignore"
```

---

## Task 5: Attribution — LICENSE and README

**Files:**
- Create: `LICENSE` (from `LICENSE.rst`)
- Create: `README.md`
- Delete: `LICENSE.rst` (replaced by `LICENSE`)

- [ ] **Step 1: Create MIT LICENSE with original copyright**

Create `LICENSE` (plain text, not rst). The original copyright notice MUST be
preserved — this is a legal requirement of the MIT license:

```text
MIT License

Copyright (c) 2019 KU Leuven Machine Learning Research Group - Tom Decroos, Pieter Robberechts
Copyright (c) 2026 Karsten S. Nielsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Create README.md**

```markdown
# silly-kicks

*The Ministry requires that all football actions be properly classified and valued.*

**silly-kicks** is a Python library for objectively quantifying the impact of
individual actions performed by football players using event stream data.

It is an independently maintained successor to
[socceraction](https://github.com/ML-KULeuven/socceraction), originally
developed by Tom Decroos and Pieter Robberechts at KU Leuven. Built under the
MIT license with full attribution preserved.

## Features

- **SPADL** — Soccer Player Action Description Language: a unified schema for
  on-ball actions with converters for StatsBomb, Wyscout, Opta, and kloppy
- **VAEP** — Valuing Actions by Estimating Probabilities: a framework for
  quantifying the value of individual actions
- **Atomic SPADL** — continuous (non-discretized) action representation

## Installation

```bash
pip install silly-kicks
```

With optional provider support:

```bash
pip install "silly-kicks[statsbomb,kloppy,xgboost]"
```

## Quick Start

```python
import silly_kicks.spadl as spadl

# Convert StatsBomb events to SPADL actions
actions = spadl.statsbomb.convert_to_actions(events, home_team_id=123)

# Add human-readable names
actions = spadl.add_names(actions)
```

## Attribution

This project builds on the foundational research by the KU Leuven Machine
Learning Research Group. If you use this library in academic work, please cite
the original papers:

```bibtex
@inproceedings{Decroos2019VAEP,
  title     = {Actions Speak Louder than Goals: Valuing Player Actions in Soccer},
  author    = {Tom Decroos and Lotte Bransen and Jan Van Haaren and Jesse Davis},
  booktitle = {Proceedings of the 25th ACM SIGKDD International Conference
               on Knowledge Discovery \& Data Mining},
  pages     = {1851--1861},
  year      = {2019},
  doi       = {10.1145/3292500.3330758}
}

@inproceedings{Decroos2020AtomicSPADL,
  title     = {Interpretable Prediction of Goals in Soccer},
  author    = {Tom Decroos and Jesse Davis},
  booktitle = {Proceedings of the AAAI-20 Workshop on Artificial Intelligence
               in Team Sports},
  year      = {2020}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
```

- [ ] **Step 3: Remove old LICENSE.rst and commit**

```bash
cd /d/Development/karstenskyt__silly-kicks
rm LICENSE.rst
git add LICENSE README.md
git rm LICENSE.rst
git commit -m "$(cat <<'EOF'
docs: add README with attribution and BibTeX citations

Original socceraction authors (Decroos, Robberechts, KU Leuven)
credited in README, LICENSE, and BibTeX entries.
MIT license preserved with dual copyright notice.
EOF
)"
```

---

## Task 6: Update silly_kicks/__init__.py Version

**Files:**
- Modify: `silly_kicks/__init__.py`

- [ ] **Step 1: Read current __init__.py**

Check current content of `silly_kicks/__init__.py`.

- [ ] **Step 2: Update version and docstring**

The file should read:

```python
"""silly-kicks: classify and value on-ball football actions.

Originally developed as socceraction by Tom Decroos and Pieter Robberechts
at KU Leuven. This is an independently maintained successor.
"""

__version__ = "0.1.0"
```

- [ ] **Step 3: Commit**

```bash
cd /d/Development/karstenskyt__silly-kicks
git add silly_kicks/__init__.py
git commit -m "chore: set version to 0.1.0, update docstring"
```

---

## Task 7: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff pyright
      - run: ruff check silly_kicks/ tests/
      - run: ruff format --check silly_kicks/ tests/
      - run: pip install -e ".[test]"
      - run: pyright silly_kicks/

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]
        exclude:
          - os: windows-latest
            python-version: "3.10"
          - os: windows-latest
            python-version: "3.11"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[statsbomb,kloppy,xgboost,test]"
      - run: pytest tests/ -m "not e2e" -v --tb=short
```

- [ ] **Step 2: Commit CI config**

```bash
cd /d/Development/karstenskyt__silly-kicks
mkdir -p .github/workflows
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow (lint + test matrix)"
```

---

## Task 8: Install and Verify Test Suite (Gate)

This is the Phase 1 gate — all existing upstream tests must pass under the new
package name with zero behavioral changes.

**Files:**
- No new files. This task verifies the work from Tasks 1-7.

- [ ] **Step 1: Create virtual environment**

```bash
cd /d/Development/karstenskyt__silly-kicks
uv venv --python 3.10 .venv
source .venv/Scripts/activate  # Windows Git Bash
```

- [ ] **Step 2: Install silly-kicks with all extras**

```bash
uv pip install -e ".[statsbomb,kloppy,xgboost,test]"
```

Expected: installs successfully with no errors.

- [ ] **Step 3: Verify the package imports correctly**

```bash
python -c "import silly_kicks; print(silly_kicks.__version__)"
```

Expected output: `0.1.0`

```bash
python -c "from silly_kicks.spadl import statsbomb, wyscout, opta, config; print('SPADL OK')"
python -c "from silly_kicks.vaep import features, labels, formula; print('VAEP OK')"
python -c "from silly_kicks.atomic.spadl import convert_to_atomic; print('Atomic OK')"
python -c "from silly_kicks.data.statsbomb import StatsBombLoader; print('Data OK')"
```

Expected: all four print their OK message.

- [ ] **Step 4: Run the full test suite (excluding e2e)**

```bash
cd /d/Development/karstenskyt__silly-kicks
python -m pytest tests/ -m "not e2e" -v --tb=short 2>&1 | head -100
```

Expected: all tests pass. If any fail, investigate — the only acceptable
failures are tests that depend on downloaded datasets (`@pytest.mark.e2e`)
which are excluded by `-m "not e2e"`.

Note: some tests may require the `statsbombpy` package to be installed for
StatsBomb data loading. If tests fail with import errors, verify all extras
were installed in Step 2.

- [ ] **Step 5: Run Ruff lint check**

```bash
cd /d/Development/karstenskyt__silly-kicks
ruff check silly_kicks/ tests/
```

Expected: this will likely report violations (upstream code doesn't follow our
rule set). **This is expected for Phase 1** — Ruff compliance is a Phase 2
deliverable. The gate for Phase 1 is that tests pass, not that lint passes.

Document the violation count for Phase 2 planning:

```bash
ruff check silly_kicks/ tests/ --statistics
```

- [ ] **Step 6: Run Pyright type check**

```bash
cd /d/Development/karstenskyt__silly-kicks
pyright silly_kicks/
```

Expected: this will likely report type errors (upstream code doesn't have full
type annotations). **This is expected for Phase 1** — Pyright compliance is a
Phase 2 deliverable.

Document the error count for Phase 2 planning.

- [ ] **Step 7: Record gate results**

After running all checks, record the results:

- Tests: PASS / FAIL (must be PASS for Phase 1 gate)
- Ruff violations: N (informational, fixed in Phase 2)
- Pyright errors: N (informational, fixed in Phase 2)

If tests PASS: Phase 1 is complete. Proceed to push and Phase 2 planning.
If tests FAIL: debug and fix before proceeding. Any fix must be a mechanical
adjustment (import path, fixture path), never a behavioral change.

---

## Task 9: Push to GitHub and Tag

**Files:**
- No new files.

- [ ] **Step 1: Set remote and push**

```bash
cd /d/Development/karstenskyt__silly-kicks
git remote add origin https://github.com/karsten-s-nielsen/silly-kicks.git
git branch -M main
git push -u origin main
```

- [ ] **Step 2: Tag the Phase 1 milestone**

```bash
git tag -a v0.1.0-alpha -m "Phase 1: clean import from socceraction v1.5.3"
git push origin v0.1.0-alpha
```

- [ ] **Step 3: Verify CI runs on GitHub**

Check `https://github.com/karsten-s-nielsen/silly-kicks/actions` — the CI
workflow from Task 7 should trigger on the push. The `test` job should pass;
the `lint` job may fail (expected — Phase 2 work).

---

## Phase 2 Planning Note

After Phase 1 gate passes, the next plan will cover Phase 2 (Modernize & Audit).
Phase 2 inputs:

- Ruff violation count from Task 8 Step 5
- Pyright error count from Task 8 Step 6
- Mad scientist audit findings (architecture, security, optimization, documentation)
- Pandera/multimethod dependency evaluation
- Decision on whether to keep or remove `data/` module (I/O code vs hex architecture)
- Decision on whether to keep or remove `xthreat.py` (separate concern from SPADL+VAEP)

Phase 2 will be planned as a separate document after these inputs are collected.
