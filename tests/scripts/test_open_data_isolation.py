"""statsbombpy's ``config.OPEN_DATA_PATHS`` is a process-global dict that
``scripts._sb_open_data.pin_open_data_revision`` edits in place. A test that pins it to a toy server
(test_dataset_read_audit runs the open-data stages in-process) must not leave it there: ``monkeypatch``
restores the env, not the dict, and every later statsbombpy call of the process then read a stopped server
(the test_shot_goalmouth_sb_e2e ConnectionRefused). The root conftest guard restores the dict after every
test. These two tests run in file order: the first pins and does not restore, the second must see the
default paths again.
"""

from __future__ import annotations

import pytest

pytest.importorskip("statsbombpy")


def test_a_pin_to_a_toy_server_without_restoring(monkeypatch):
    from statsbombpy import config  # type: ignore[import-not-found]

    import scripts._sb_open_data as O

    monkeypatch.setenv(O.OPEN_DATA_BASE_ENV, "http://127.0.0.1:9")
    O.pin_open_data_revision("0" * 40)
    assert any("127.0.0.1:9" in u for u in config.OPEN_DATA_PATHS.values())  # non-vacuity: the dict really moved


def test_b_the_next_test_sees_the_default_paths():
    from statsbombpy import config  # type: ignore[import-not-found]

    assert not any("127.0.0.1" in u for u in config.OPEN_DATA_PATHS.values()), config.OPEN_DATA_PATHS
