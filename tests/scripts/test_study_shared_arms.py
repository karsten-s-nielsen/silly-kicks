import numpy as np
import pandas as pd

from scripts._study_shared import load_study_inputs, persist_study_inputs


def test_arms_round_trip(tmp_path):
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    y = np.array([0, 1, 0])
    groups = np.array(["g1", "g2", "g3"])
    providers = np.array(["sportec", "skillcorner", "skillcorner"])
    match_ids = np.array(["DFL-MAT-J03WMX", "1874553", "9999"])
    is_public = np.array([True, True, False])
    arms = {"public": is_public, "sc_extended": np.array([True, True, True])}
    persist_study_inputs(
        tmp_path,
        X=X,
        y=y,
        groups=groups,
        providers=providers,
        match_ids=match_ids,
        is_public=is_public,
        config={"run_paired": True},
        arms=arms,
    )
    inp = load_study_inputs(tmp_path)
    assert inp.arms["public"].tolist() == [True, True, False]
    assert inp.arms["sc_extended"].tolist() == [True, True, True]
    assert inp.is_public.tolist() == [True, True, False]


def test_arms_default_empty_when_absent(tmp_path):
    # back-compat: an inputs dir written without arms loads with arms == {}
    X = pd.DataFrame({"f": [0.0, 1.0]})
    persist_study_inputs(
        tmp_path,
        X=X,
        y=np.array([0, 1]),
        groups=np.array(["g1", "g2"]),
        providers=np.array(["skillcorner", "skillcorner"]),
        match_ids=np.array(["1", "2"]),
        is_public=np.array([True, True]),
        config={},
    )
    assert load_study_inputs(tmp_path).arms == {}
