import pandas as pd
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
