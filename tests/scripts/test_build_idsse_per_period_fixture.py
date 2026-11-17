"""R64 / bar A26: the idsse full-match fixture is built from the raw public DFL files by this library's parser.

- ``scripts/build_idsse_per_period_fixture.py`` writes the parser's bronze as it is (hermetic test on the committed
  slice ``tests/datasets/sportec/idsse_slice/``).
- ``scripts/extract_provider_fixtures.py`` refuses its retired idsse ``per_period`` path at the argument check,
  before it reads any Databricks variable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import build_idsse_per_period_fixture as gen
from scripts import extract_provider_fixtures as old
from silly_kicks.providers.sportec import parse_dfl_events, parse_dfl_match_info

_SLICE = Path(__file__).resolve().parents[1] / "datasets" / "sportec" / "idsse_slice"


def test_generator_writes_the_parsers_bronze_as_it_is(tmp_path):
    out = tmp_path / "per_period_match.parquet"
    args = ["--match-info", str(_SLICE / "info.xml"), "--events", str(_SLICE / "events.xml"), "--out", str(out)]
    assert gen.main(args) == 0
    built = pd.read_parquet(out)
    want = parse_dfl_events(str(_SLICE / "events.xml"), match_info=parse_dfl_match_info(str(_SLICE / "info.xml")),
                            match_id="J03WMX")  # fmt: skip
    assert len(built) > 0  # non-vacuity
    pd.testing.assert_frame_equal(built, want.reset_index(drop=True))
    assert set(built["match_id"]) == {"J03WMX"}  # the parser's form; no step after the parse


def test_generator_refuses_without_the_raw_files():
    with pytest.raises(SystemExit):
        gen.main([])


@pytest.mark.parametrize("provider", ["idsse", "all"])
def test_old_script_refuses_the_idsse_per_period_path_before_any_work(provider, monkeypatch, capsys):
    for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_HTTP_PATH"):
        monkeypatch.delenv(var, raising=False)
    called = []
    monkeypatch.setattr(old, "_extract_idsse", lambda *a, **k: called.append("idsse"))
    monkeypatch.setattr(old, "_extract_metrica_per_period", lambda *a, **k: called.append("metrica"))
    code = old.main(["--provider", provider, "--variant", "per_period"])
    err = capsys.readouterr().err
    assert code != 0
    assert "build_idsse_per_period_fixture.py" in err
    assert "missing env vars" not in err  # refused at the argument check, not at the Databricks connect
    assert called == []  # no work at all, so no file is written and then fails
    if provider == "all":
        assert "--provider metrica --variant per_period" in err
