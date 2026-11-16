"""The staged-equals-served check (scorer spec S4): staged frames must equal the live loader's frames on every
detector input column and on the row set, exactly; ids compare by value (a stored categorical team id is the same
id)."""

from __future__ import annotations

from tests.scripts.test_materialize_gs_contact_truth import _frames, _source


def _staged():
    s = _frames("3816").assign(source_provider="gradientsports")
    s["team_id"] = s["team_id"].astype(object)  # the stored form: a plain id column
    return s


def test_identical_frames_pass():
    from scripts.check_gs_contact_truth import compare_frames

    assert compare_frames(_staged(), _frames("3816")) == []


def test_shifted_clock_fails():
    from scripts.check_gs_contact_truth import compare_frames

    shifted = _staged().assign(time_seconds=lambda f: f["time_seconds"] + 2700.0)
    assert compare_frames(shifted, _frames("3816")) == ["time_seconds"]


def test_missing_z_source_fails():
    from scripts.check_gs_contact_truth import compare_frames

    assert compare_frames(_staged().drop(columns="z_source"), _frames("3816")) == ["z_source"]


def test_a_dropped_row_fails():
    from scripts.check_gs_contact_truth import compare_frames

    assert compare_frames(_staged().iloc[:-1], _frames("3816")) == ["rows"]


def test_a_changed_position_fails():
    from scripts.check_gs_contact_truth import compare_frames

    moved = _staged()
    moved.loc[0, "x"] = 50.01
    assert compare_frames(moved, _frames("3816")) == ["x"]


def test_run_flags_only_the_unit_that_differs(tmp_path):
    import scripts.check_gs_contact_truth as c
    import scripts.materialize_gs_contact_truth as m

    prov = {"commit": "c4", "dirty": False}
    m.run(tmp_path / "data", work_dir=tmp_path / "w1", provenance=prov, source=_source())
    refs, load = _source()

    def live(ref):
        lm = load(ref)[0]
        frames = lm[3].assign(x=lm[3]["x"] + (0.5 if ref.match_id == "3812" else 0.0))
        return (lm[0], lm[1], lm[2], frames, lm[4])

    rep = c.run(tmp_path / "data", work_dir=tmp_path / "w2", provenance=prov, source=(refs, live))
    assert rep["n_units"] == 2 and rep["mismatches"] == {"gradientsports-3812": ["x"]}


def test_the_comparison_is_exact_and_covers_team_id():
    from scripts.check_gs_contact_truth import compare_frames

    tiny = _staged()
    tiny.loc[1, "y"] = 34.0 + 1e-9  # the staged y is 34.0; no tolerance: any difference fails
    assert compare_frames(tiny, _frames("3816")) == ["y"]
    other = _staged()
    other.loc[1, "team_id"] = 6
    assert compare_frames(other, _frames("3816")) == ["team_id"]
