"""The gs_contact_truth materializer (scorer spec S4): one pergame unit per GS match with the frames as the loader
returns them, the SPADL actions and the PFF on-ball sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

_EV = [
    {"startTime": 100.0, "gameEvents": {"gameEventType": "FIRSTKICKOFF", "period": 1}},
    {
        "startTime": 101.0,
        "endTime": 102.5,
        "gameEvents": {"gameEventType": "OTB", "period": 1, "playerId": 7},
        "possessionEvents": {"possessionEventType": "PA"},
    },
]


@dataclass
class _Ref:
    match_id: str
    provider: str = "gradientsports"

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.match_id)


def _frames(game: str) -> pd.DataFrame:
    f = pd.DataFrame(
        {
            "game_id": [int(game)] * 3,
            "period_id": [1, 1, 1],
            "frame_id": [1, 1, 2],
            "time_seconds": [0.0, 0.0, 0.04],
            "player_id": pd.array([pd.NA, 7, pd.NA], dtype="Int64"),
            "team_id": pd.Categorical([pd.NA, 5, pd.NA]),
            "is_ball": [True, False, True],
            "x": [50.0, 49.0, 50.5],
            "y": [34.0, 34.0, 34.0],
            "z": [0.1, 0.0, 0.1],
            "z_source": ["measured", "absent", "measured"],
            "frame_rate": [25.0] * 3,
        }
    )
    return f


def _source():
    refs = [_Ref("3816"), _Ref("3812")]
    actions = pd.DataFrame({"game_id": [1], "player_id": pd.array([7], dtype="Int64"), "team_id": [5.0]})

    def load(ref):
        return ("gradientsports", ref.match_id, actions, _frames(ref.match_id), None), _EV

    return refs, load


def test_writes_one_pergame_unit_per_match(tmp_path):
    import scripts.materialize_gs_contact_truth as m

    out = tmp_path / "gs_contact_truth"
    m.run(out, work_dir=tmp_path / "work", provenance={"commit": "c4", "dirty": False}, source=_source())
    units = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert units == ["gradientsports-3812", "gradientsports-3816"]
    for u in units:
        assert sorted(p.name for p in (out / u).iterdir()) == ["actions.parquet", "frames.parquet", "sequences.parquet"]
    frames = pd.read_parquet(out / "gradientsports-3816" / "frames.parquet")
    assert set(frames["source_provider"]) == {"gradientsports"}  # what `record` reads for the provider
    assert not isinstance(frames["team_id"].dtype, pd.CategoricalDtype)  # a plain id column for `id-kinds`
    import pyarrow as pa
    import pyarrow.parquet as pq

    stored = pq.read_schema(out / "gradientsports-3816" / "frames.parquet").field("team_id").type
    assert stored == pa.float64()  # stored as values, not an Arrow dictionary (a categorical reads back as float)
    live = _frames("3816")
    pd.testing.assert_series_equal(frames["time_seconds"], live["time_seconds"])  # the loader's own clock
    assert frames["z_source"].tolist() == live["z_source"].tolist()
    seq = pd.read_parquet(out / "gradientsports-3816" / "sequences.parquet")
    assert seq[["period_id", "t0", "t1", "player", "ptype"]].values.tolist() == [[1, 1.0, 2.5, "7", "PA"]]
    stamp = json.loads((out / "_materialize.json").read_text(encoding="utf-8"))
    assert stamp["run_commit"] == "c4" and stamp["run_tree_dirty"] is False and stamp["n_units"] == 2


def test_rerun_resumes(tmp_path):
    import scripts.materialize_gs_contact_truth as m

    calls = []
    refs, load = _source()

    def counting(ref):
        calls.append(ref.match_id)
        return load(ref)

    kw = {"work_dir": tmp_path / "work", "provenance": {"commit": "c4", "dirty": False}}
    m.run(tmp_path / "out", source=(refs, counting), **kw)
    m.run(tmp_path / "out", source=(refs, counting), **kw)
    assert sorted(calls) == ["3812", "3816"]  # the second run loads nothing
