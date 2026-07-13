"""ADR-019: extract_xshot_features must be invariant to id dtype (Int64 vs str team ids).
The xCross extractor was canonicalized in 4.18.0; xS still used raw == (spec §4)."""

import pandas as pd
import pandas.testing as pdt

from silly_kicks.tracking._xshot_occurrence import extract_xshot_features
from tests.tracking._probe_fixtures import probe_frames


def test_extract_xshot_features_invariant_to_team_id_dtype():
    frames = probe_frames()
    grp = frames[frames["frame_id"] == frames["frame_id"].iloc[0]].reset_index(drop=True)
    grp_str = grp.copy()
    # numeric-vs-string worst case: '366.0' != '366' was the shipped GS bug class
    grp_num = grp.copy()
    grp_num["team_id"] = pd.array([1 if t == "A" else 2 for t in grp["team_id"]], dtype="Int64")
    f_str = extract_xshot_features(grp_str, gk_team_id="B", goal_x=105.0)
    f_num = extract_xshot_features(grp_num, gk_team_id=2, goal_x=105.0)
    pdt.assert_frame_equal(f_str, f_num)
    # cross-dtype: numeric frames, string gk_team_id -- must not silently empty the GK mask
    f_cross = extract_xshot_features(grp_num, gk_team_id="2", goal_x=105.0)
    pdt.assert_frame_equal(f_cross, f_num)
