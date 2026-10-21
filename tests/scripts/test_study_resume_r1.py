"""Decision R1 (spec 5.3): does a study whose store is RESUMED after an interruption reproduce the
uninterrupted study?

Three SUBPROCESSES, each with fresh sampler state exactly as a relaunched job has: A = 6 trials
uninterrupted in store ``a/``; B1 = 3 trials in store ``b/`` (the interrupted attempt); B2 = 6 trials in
the SAME store ``b/`` (the resume). The observed outcome is pinned by ``STUDY_RESUME`` in
``scripts/train_xshot_occurrence.py``: ``True`` asserts equality, ``False`` asserts inequality -- so a
future ``ruthless`` that changes the answer turns this red and prompts the switch.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("ruthless")

REPO = pathlib.Path(__file__).resolve().parents[2]

_CHILD = """
import json, pathlib, sys
import numpy as np
import pandas as pd
from scripts.train_xshot_occurrence import _hpo_once

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
y = (X["a"] + rng.normal(scale=0.5, size=300) > 1.0).astype(int).to_numpy()
groups = np.repeat(np.arange(15), 20).astype(str)
store = pathlib.Path(sys.argv[1])
store.mkdir(parents=True, exist_ok=True)
params = _hpo_once(
    X, y, groups, store, "r1", int(sys.argv[2]),
    objective_inputs={"driver": "test_study_resume_r1"},
    prov={"commit": "r1", "dirty": False, "tree_state": "clean"},
)
print(json.dumps(params, sort_keys=True))
"""


def _study(store: pathlib.Path, n_trials: int) -> dict:
    out = subprocess.run(  # noqa: S603 -- fixed argv: this interpreter + an inline child script
        [sys.executable, "-c", _CHILD, str(store), str(n_trials)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.slow  # slow: measured 25 s on the Windows dev box (ADR-023)
def test_a_resumed_study_matches_the_recorded_R1_decision(tmp_path):
    from scripts.train_xshot_occurrence import STUDY_RESUME

    uninterrupted = _study(tmp_path / "a", 6)
    _study(tmp_path / "b", 3)
    resumed = _study(tmp_path / "b", 6)
    assert any((tmp_path / "b").iterdir()), "non-vacuity: the interrupted attempt left a store to resume"
    if STUDY_RESUME:
        assert resumed == uninterrupted, "STUDY_RESUME=True but a resumed study diverged"
    else:
        assert resumed != uninterrupted, (
            "a resumed study now reproduces the uninterrupted one: re-run R1 and switch STUDY_RESUME to True"
        )
