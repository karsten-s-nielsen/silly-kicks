import pandas as pd

from scripts.hf_jobs.repackage import flat_to_pergame


def _write_flat(root):
    # Flat TC3 shape: shards/<gen>/<provider>__<game>.parquet (frames) + _actions/<provider>__<game>.parquet.
    gen = root / "shards" / "ac98a040f3c2ec8c"
    gen.mkdir(parents=True)
    (root / "_actions").mkdir()
    pd.DataFrame({"source_provider": ["gradientsports"], "game_id": ["10502"]}).to_parquet(
        gen / "gradientsports__10502.parquet"
    )
    # type_id 11 = shot (kept); 0/21 are non-shot (dropped). game_id/period_id/team_id/time_seconds pass through.
    pd.DataFrame(
        {
            "game_id": ["10502"] * 3,
            "period_id": [1, 1, 2],
            "team_id": ["A", "A", "B"],
            "time_seconds": [10.0, 20.0, 30.0],
            "type_id": [11, 0, 21],
        }
    ).to_parquet(root / "_actions" / "gradientsports__10502.parquet")


def test_repackage_makes_per_game_dirs_with_bare_names_and_filtered_shots(tmp_path):
    flat, out = tmp_path / "corpus", tmp_path / "pergame"
    flat.mkdir()
    _write_flat(flat)

    n = flat_to_pergame(flat, out)
    assert n == 1
    # bare, no "__" (the scripts/_driver key separator rejects it): "<provider>-<game>"
    gdir = out / "gradientsports-10502"
    assert (gdir / "frames.parquet").exists()
    shots = pd.read_parquet(gdir / "shots.parquet")
    assert list(shots["type_id"]) == [11]  # only the shot-type action survived
    assert set(shots["type_name"]) == {"shot"}  # type_name added from type_id
    assert {"game_id", "period_id", "team_id", "time_seconds"} <= set(shots.columns)


def test_repackage_tolerates_a_match_with_no_actions(tmp_path):
    flat, out = tmp_path / "corpus", tmp_path / "pergame"
    gen = flat / "shards" / "g"
    gen.mkdir(parents=True)
    pd.DataFrame({"source_provider": ["idsse"], "game_id": ["X1"]}).to_parquet(gen / "idsse__X1.parquet")
    n = flat_to_pergame(flat, out)
    assert n == 1
    shots = pd.read_parquet(out / "idsse-X1" / "shots.parquet")
    assert len(shots) == 0  # empty shots (no _actions) -> an empty shards.parquet, not a crash
