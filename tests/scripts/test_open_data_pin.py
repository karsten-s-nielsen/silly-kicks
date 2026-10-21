"""Decision P6: StatsBomb open data is read at ONE pinned commit."""

from __future__ import annotations

import pytest

statsbombpy = pytest.importorskip("statsbombpy")

from scripts import _sb_open_data as O  # noqa: E402

REV = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"


def test_pin_rewrites_every_open_data_url():
    from statsbombpy import config  # type: ignore[import-not-found]

    saved = dict(config.OPEN_DATA_PATHS)
    try:
        assert O.pin_open_data_revision(REV) == REV
        assert all(f"/open-data/{REV}/data/" in u for u in config.OPEN_DATA_PATHS.values())
        assert O.pin_open_data_revision(REV) == REV  # idempotent
        assert all(f"/open-data/{REV}/data/" in u for u in config.OPEN_DATA_PATHS.values())
    finally:
        config.OPEN_DATA_PATHS.clear()
        config.OPEN_DATA_PATHS.update(saved)


def test_pin_reads_the_environment(monkeypatch):
    from statsbombpy import config  # type: ignore[import-not-found]

    saved = dict(config.OPEN_DATA_PATHS)
    monkeypatch.setenv(O.OPEN_DATA_REVISION_ENV, REV)
    try:
        O.assert_statsbomb_open_data_mode()
        assert all(REV in u for u in config.OPEN_DATA_PATHS.values())
    finally:
        config.OPEN_DATA_PATHS.clear()
        config.OPEN_DATA_PATHS.update(saved)


def test_unset_environment_leaves_the_urls_alone(monkeypatch):
    from statsbombpy import config  # type: ignore[import-not-found]

    saved = dict(config.OPEN_DATA_PATHS)
    monkeypatch.delenv(O.OPEN_DATA_REVISION_ENV, raising=False)
    monkeypatch.delenv(O.OPEN_DATA_BASE_ENV, raising=False)
    O.assert_statsbomb_open_data_mode()
    assert dict(config.OPEN_DATA_PATHS) == saved


def test_a_non_sha_revision_is_refused():
    with pytest.raises(ValueError, match="40-hex"):
        O.pin_open_data_revision("master")
