"""TF-48 atomic mirror: thin delegation (engine consumes NO action coordinates);
atomic shot domain is {shot, shot_penalty} -- shot_freekick is a `freekick` atom
(atomic/spadl/base.py, existing pre-shot-GK precedent)."""

import pandas as pd
import pandas.testing as pdt

from silly_kicks.atomic.spadl import config as atomicconfig
from silly_kicks.atomic.tracking.features import add_shot_goalmouth as atomic_add
from silly_kicks.tracking.features import add_shot_goalmouth as std_add
from tests.tracking.test_shot_goalmouth import make_match


def _to_atomic(actions: pd.DataFrame) -> pd.DataFrame:
    a = actions.copy()
    a["x"], a["y"] = a.pop("start_x"), a.pop("start_y")
    a["dx"] = actions["end_x"] - actions["start_x"]
    a["dy"] = actions["end_y"] - actions["start_y"]
    a = a.drop(columns=["end_x", "end_y", "result_id"])
    a["type_id"] = atomicconfig.actiontype_id["shot"]
    return a


def test_parity_with_standard_on_shot_rows():
    actions, frames = make_match()
    std = std_add(actions, frames)
    atm = atomic_add(_to_atomic(actions), frames)
    cols = [c for c in std.columns if c.startswith("shot_")]
    pdt.assert_frame_equal(std[cols].reset_index(drop=True), atm[cols].reset_index(drop=True))


def test_atomic_domain_excludes_freekick_atoms():
    actions, frames = make_match()
    a = _to_atomic(actions)
    a["type_id"] = atomicconfig.actiontype_id["freekick"]  # direct FK shot in atomic space
    out = atomic_add(a, frames)
    assert out["shot_crossing_source"].isna().all()


def test_atomic_per_series_wrapper():
    actions, frames = make_match()
    from silly_kicks.atomic.tracking.features import shot_crossing_y

    s = shot_crossing_y(_to_atomic(actions), frames)
    assert isinstance(s, pd.Series) and s.name == "shot_crossing_y"
