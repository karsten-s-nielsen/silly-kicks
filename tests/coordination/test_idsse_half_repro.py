"""TF-58 Task 18 Step 5 (ADR-056): the committed idsse_half fixture reproduces byte-for-byte.

Marked ``e2e`` because it loads the real DFL match through the pining loader (needs ``PINING_FOR_THE_DATA_TOKEN``
and network); the committed parquets and the fixture README are what CI reads.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).resolve().parents[1] / "datasets" / "tracking" / "idsse_half"


def test_fixture_readme_states_the_committed_sizes():
    # the README's Contents table is what a reader sees without the token; it must not go stale (review minor 20).
    readme = (_FIXTURE / "README.md").read_text(encoding="utf-8")
    stated = dict(re.findall(r"^\| `(\w+\.parquet)` \| [\d,]+ \| ([\d.]+) MB \|", readme, flags=re.M))
    assert set(stated) == {"frames.parquet", "actions.parquet"}  # non-vacuity: both rows parsed
    for name, mb in stated.items():
        assert mb == f"{(_FIXTURE / name).stat().st_size / 1e6:.2f}", name


def _footer_writer(path: Path) -> dict[str, str]:
    """The writer a parquet's footer records: pandas (its ``pandas`` metadata) and pyarrow (``created_by``)."""
    import json

    import pyarrow.parquet as pq

    meta = pq.ParquetFile(path).metadata
    pandas_meta = json.loads((meta.metadata or {})[b"pandas"].decode())
    return {"pandas": pandas_meta["pandas_version"], "pyarrow": meta.created_by.rsplit(" ", 1)[-1]}


def _stated_writer() -> dict[str, str]:
    readme = (_FIXTURE / "README.md").read_text(encoding="utf-8")
    found = re.search(r"written with pandas `([^`]+)` and pyarrow `([^`]+)`", readme)
    assert found, "the README must state the writer environment"
    return {"pandas": found.group(1), "pyarrow": found.group(2)}


def test_fixture_readme_states_the_writer_environment():
    # Parquet footers record the writer's pandas and pyarrow versions, so the committed bytes reproduce only under
    # that writer. The README states it, and both committed files must agree with it (a fixture regenerated under
    # another writer must restate it).
    stated = _stated_writer()
    for name in ("frames.parquet", "actions.parquet"):
        assert _footer_writer(_FIXTURE / name) == stated, name


@pytest.mark.e2e
def test_idsse_half_generator_reproduces_fixture(tmp_path):
    import pandas as pd
    import pyarrow

    from tests.datasets.tracking.idsse_half.generate_fixture import HERE, _write, build_fixture

    running = {"pandas": pd.__version__, "pyarrow": pyarrow.__version__}
    assert running == _stated_writer(), (
        f"byte-for-byte reproduction needs the fixture's writer {_stated_writer()} (the parquet footer records it); "
        f"this interpreter has {running}"
    )
    frames, actions = build_fixture()
    for name, df in (("frames", frames), ("actions", actions)):
        fresh = tmp_path / f"{name}.parquet"
        _write(df, fresh)
        committed = (HERE / f"{name}.parquet").read_bytes()
        assert fresh.read_bytes() == committed, f"{name}.parquet is not byte-identical to the committed fixture"
