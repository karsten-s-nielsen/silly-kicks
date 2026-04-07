import numpy as np
import pandas as pd

import silly_kicks.atomic.spadl.utils as aspu
import silly_kicks.atomic.vaep.labels as alab
import silly_kicks.spadl.utils as spu
import silly_kicks.vaep.labels as lab


def test_scores(spadl_actions: pd.DataFrame) -> None:
    nr_actions = 10
    spadl_actions = spu.add_names(spadl_actions)
    scores = lab.scores(spadl_actions, nr_actions)
    assert len(scores) == len(spadl_actions)


def test_conceds(spadl_actions: pd.DataFrame) -> None:
    nr_actions = 10
    spadl_actions = spu.add_names(spadl_actions)
    concedes = lab.concedes(spadl_actions, nr_actions)
    assert len(concedes) == len(spadl_actions)


def test_atomic_scores(atomic_spadl_actions: pd.DataFrame) -> None:
    atomic_spadl_actions = aspu.add_names(atomic_spadl_actions)
    scores = alab.scores(atomic_spadl_actions, 10)
    assert len(scores) == len(atomic_spadl_actions)
    assert scores.columns.tolist() == ["scores"]
    assert scores.dtypes["scores"] is np.dtype(bool)


def test_atomic_concedes(atomic_spadl_actions: pd.DataFrame) -> None:
    atomic_spadl_actions = aspu.add_names(atomic_spadl_actions)
    concedes = alab.concedes(atomic_spadl_actions, 10)
    assert len(concedes) == len(atomic_spadl_actions)
    assert concedes.columns.tolist() == ["concedes"]


def test_scores_dtype(spadl_actions: pd.DataFrame) -> None:
    spadl_actions = spu.add_names(spadl_actions)
    scores = lab.scores(spadl_actions, 10)
    assert scores.dtypes["scores"] is np.dtype(bool)


def test_concedes_dtype(spadl_actions: pd.DataFrame) -> None:
    spadl_actions = spu.add_names(spadl_actions)
    concedes = lab.concedes(spadl_actions, 10)
    assert concedes.dtypes["concedes"] is np.dtype(bool)


def test_scores_xg(spadl_actions: pd.DataFrame) -> None:
    """xG-targeted labels should produce float scores."""
    spadl_actions = spu.add_names(spadl_actions)
    spadl_actions = spadl_actions.copy()
    spadl_actions["xg"] = 0.0
    shot_mask = spadl_actions["type_name"].str.contains("shot")
    spadl_actions.loc[shot_mask, "xg"] = 0.5
    scores = lab.scores(spadl_actions, 10, xg_column="xg")
    assert len(scores) == len(spadl_actions)
    assert scores["scores"].dtype == float


def test_save_from_shot(spadl_actions: pd.DataFrame) -> None:
    spadl_actions = spu.add_names(spadl_actions)
    result = lab.save_from_shot(spadl_actions)
    assert len(result) == len(spadl_actions)
    assert result.columns.tolist() == ["save_from_shot"]


def test_claim_from_cross(spadl_actions: pd.DataFrame) -> None:
    spadl_actions = spu.add_names(spadl_actions)
    result = lab.claim_from_cross(spadl_actions)
    assert len(result) == len(spadl_actions)
    assert result.columns.tolist() == ["claim_from_cross"]


def test_scores_backward_compat(spadl_actions: pd.DataFrame) -> None:
    """scores() without xg_column should produce binary labels."""
    spadl_actions = spu.add_names(spadl_actions)
    scores = lab.scores(spadl_actions, 10)
    assert scores["scores"].dtype == bool
