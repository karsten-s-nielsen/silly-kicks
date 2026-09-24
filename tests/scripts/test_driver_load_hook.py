"""`for_each(load=)` resume-before-load, and first-class exclusion markers (spec §4.3).

tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
from _driver import (
    already_excluded,
    exclusion_path,
    for_each,
    generation_dir,
    join_key,
    read_exclusion,
    shard_path,
    write_exclusion,
)
from _item_outcome import ItemExcluded

_TOKEN = {"schema": "load-hook-test-1"}


def _gen(tmp_path) -> pathlib.Path:
    return generation_dir(tmp_path / "sh", token_inputs=_TOKEN)


# ---- Step 2: marker helpers -----------------------------------------------------------------------


def test_exclusion_path_is_json_beside_the_shard(tmp_path):
    gen = _gen(tmp_path)
    p = exclusion_path(gen, ("skillcorner", "m1"))
    assert p == pathlib.Path(gen) / f"{join_key(('skillcorner', 'm1'))}.excluded.json"
    assert p.suffix == ".json"


def test_write_then_read_round_trips_reason_and_details(tmp_path):
    gen = _gen(tmp_path)
    key = ("skillcorner", "m1")
    assert not already_excluded(gen, key)
    write_exclusion(gen, key, ItemExcluded("S1 gate", details={"ball_off_pitch_rate": 0.02}))
    assert already_excluded(gen, key)
    assert read_exclusion(gen, key) == {"reason": "S1 gate", "details": {"ball_off_pitch_rate": 0.02}}


def test_read_exclusion_of_absent_marker_is_none(tmp_path):
    assert read_exclusion(_gen(tmp_path), ("skillcorner", "nope")) is None


def test_marker_is_invisible_to_parquet_glob(tmp_path):
    gen = _gen(tmp_path)
    write_exclusion(gen, ("skillcorner", "m1"), ItemExcluded("S1 gate"))
    assert list(pathlib.Path(gen).glob("*.parquet")) == []


def test_a_truncated_marker_reads_as_none_and_warns(tmp_path):
    gen = _gen(tmp_path)
    key = ("skillcorner", "m1")
    exclusion_path(gen, key).write_text("{not json", encoding="utf-8")
    with pytest.warns(UserWarning):
        assert read_exclusion(gen, key) is None


# ---- Steps 3-5: for_each(load=) resume-before-load, exclusion outcome, manifest ------------------


class _Spy:
    """A `load=` stand-in: records every ref it is asked to load; serves a 1-row frame's worth."""

    def __init__(self, *, exclude=(), fail=()):
        self.exclude = set(exclude)
        self.fail = set(fail)
        self.calls: list = []

    def __call__(self, ref):
        self.calls.append(ref)
        if ref in self.exclude:
            raise ItemExcluded(f"excluded {ref}")
        if ref in self.fail:
            raise RuntimeError(f"boom {ref}")
        return ref


def _rows_of(item) -> pd.DataFrame:
    return pd.DataFrame({"ref": [str(item)]})


def _run(tmp_path, refs, *, load, work=_rows_of, counters=None, max_consecutive_failures=3):
    return for_each(
        refs,
        key=lambda r: r if isinstance(r, tuple) else (str(r),),
        work=work,
        shard_root=tmp_path / "sh",
        token_inputs=_TOKEN,
        load=load,
        counters=counters,
        max_consecutive_failures=max_consecutive_failures,
    )


def test_load_is_called_for_a_fresh_key_and_NOT_for_a_finished_one(tmp_path):
    spy = _Spy()
    first = _run(tmp_path, ["m1", "m2"], load=spy)
    assert spy.calls == ["m1", "m2"]  # non-vacuous: the fresh keys DID load
    assert first.attempted == 2 and first.skipped == 0

    second = _run(tmp_path, ["m1", "m2"], load=spy)
    assert spy.calls == ["m1", "m2"], "a finished key was re-loaded"  # unchanged: nothing new loaded
    assert second.skipped == 2 and second.attempted == 0


def test_load_is_NOT_called_for_a_marked_key_on_resume(tmp_path):
    spy = _Spy(exclude=["m1"])
    first = _run(tmp_path, ["m1", "m2"], load=spy)
    assert first.excluded == 1 and first.exclusions == {"m1": "excluded m1"}
    assert first.attempted == 2  # a FRESH exclusion counts as attempted (interp 10)

    spy.calls.clear()
    second = _run(tmp_path, ["m1", "m2"], load=spy)
    assert spy.calls == [], "a replayed exclusion (m1) or a finished shard (m2) re-loaded"
    assert second.excluded == 1 and second.attempted == 0 and second.skipped == 1


def test_exclusion_from_load_is_a_marker_invisible_to_parquet(tmp_path):
    spy = _Spy(exclude=["m1"])
    res = _run(tmp_path, ["m1"], load=spy)
    gen = res.shard_dir
    assert already_excluded(gen, ("m1",)) and not shard_path(gen, ("m1",)).is_file()
    assert list(pathlib.Path(gen).glob("*.parquet")) == []
    assert res.shard_keys == ()  # excluded key is not a shard key


def test_exclusion_from_work_writes_a_marker(tmp_path):
    def _work(item):
        raise ItemExcluded(f"work-excluded {item}")

    res = _run(tmp_path, ["m1"], load=lambda r: r, work=_work)
    assert res.exclusions == {"m1": "work-excluded m1"}
    assert already_excluded(res.shard_dir, ("m1",))


def test_an_exclusion_resets_the_consecutive_failure_run(tmp_path):
    # fail, fail, EXCLUDE (resets), fail -> run never reaches 3, so no abort.
    spy = _Spy(fail=["f1", "f2", "f3"], exclude=["ex"])
    res = _run(tmp_path, ["f1", "f2", "ex", "f3", "ok"], load=spy)
    assert res.failed == 3 and res.excluded == 1
    assert shard_path(res.shard_dir, ("ok",)).is_file()


def test_three_consecutive_real_failures_still_abort(tmp_path):
    spy = _Spy(fail=["f1", "f2", "f3"])
    with pytest.raises(RuntimeError, match="consecutive failures"):
        _run(tmp_path, ["f1", "f2", "f3", "f4"], load=spy)


def test_conservation_passes_with_an_exclusion(tmp_path):
    spy = _Spy(exclude=["m2"])
    res = _run(tmp_path, ["m1", "m2", "m3"], load=spy)  # would raise inside for_each if conservation failed
    assert res.attempted == 3 and res.excluded == 1 and res.failed == 0
    assert set(res.shard_keys) == {"m1", "m3"}


def test_manifest_carries_n_excluded(tmp_path):
    spy = _Spy(exclude=["m2"])
    res = _run(tmp_path, ["m1", "m2"], load=spy)
    assert res.manifest()["n_excluded"] == 1


def test_manifest_fields_requires_excluded_keyword():
    from _driver import manifest_fields

    with pytest.raises(TypeError):
        manifest_fields("gen", attempted=1, failed=0, counters_unrecorded=0)  # type: ignore[call-arg]  # missing excluded=


def test_no_load_pass_is_unchanged_plus_n_excluded_zero(tmp_path):
    """A pass WITHOUT `load=` behaves exactly as before, plus `n_excluded == 0` (interp 3): the field
    is present on EVERY manifest, and the rest of the manifest is unchanged."""
    res = for_each(["a", "b"], key=lambda r: (r,), work=_rows_of, shard_root=tmp_path / "sh", token_inputs=_TOKEN)
    assert res.excluded == 0 and res.exclusions == {}
    assert res.shard_keys == ("a", "b")
    m = res.manifest()
    assert m["n_excluded"] == 0
    assert set(m) == {"generation", "n_attempted", "n_failed", "n_counters_unrecorded", "n_excluded"}
