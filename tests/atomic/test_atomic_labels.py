import pandas as pd

import silly_kicks.atomic.spadl.utils as spu
import silly_kicks.atomic.vaep.labels as lab


def test_scores(atomic_spadl_actions: pd.DataFrame) -> None:
    nr_actions = 10
    atomic_spadl_actions = spu.add_names(atomic_spadl_actions)
    scores = lab.scores(atomic_spadl_actions, nr_actions)
    assert len(scores) == len(atomic_spadl_actions)


def test_conceds(atomic_spadl_actions: pd.DataFrame) -> None:
    nr_actions = 10
    atomic_spadl_actions = spu.add_names(atomic_spadl_actions)
    concedes = lab.concedes(atomic_spadl_actions, nr_actions)
    assert len(concedes) == len(atomic_spadl_actions)
