"""CI-safe: `_loader_databricks._connect` selects PAT vs OAuth auth correctly.

No real Databricks and no real connector -- fake `databricks.sql` / `databricks.sdk.core` are
injected so ONLY the auth-branch logic is exercised (the connection itself is a stand-in object).
"""

from __future__ import annotations

import sys
import types

import _loader_databricks as ldb
import pytest


class _FakeConnect:
    """Records the kwargs `_connect` passes to `databricks.sql.connect`."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return object()  # stand-in "connection"


@pytest.fixture
def fake_dbsql(monkeypatch):
    conn = _FakeConnect()
    mod = types.ModuleType("databricks.sql")
    mod.connect = conn  # type: ignore[attr-defined]
    pkg = types.ModuleType("databricks")
    pkg.sql = mod  # type: ignore[attr-defined]  # so `import databricks.sql as dbsql` resolves
    monkeypatch.setitem(sys.modules, "databricks", pkg)
    monkeypatch.setitem(sys.modules, "databricks.sql", mod)
    return conn


@pytest.fixture
def fake_sdk(monkeypatch):
    """Inject a fake `databricks.sdk.core.Config`; returns the dict capturing the profile used."""
    captured: dict = {}

    class _FakeConfig:
        def __init__(self, *, profile):
            captured["profile"] = profile
            self.host = "https://oauth.example.databricks.com"

        def authenticate(self):  # the sql connector calls credentials_provider()() -> headers
            return {}

    core = types.ModuleType("databricks.sdk.core")
    core.Config = _FakeConfig  # type: ignore[attr-defined]
    sdk = types.ModuleType("databricks.sdk")
    sdk.core = core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk)
    monkeypatch.setitem(sys.modules, "databricks.sdk.core", core)
    return captured


def test_pat_path_uses_access_token(monkeypatch, fake_dbsql):
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapiXXX")
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    ldb._connect()
    kw = fake_dbsql.kwargs
    assert kw["access_token"] == "dapiXXX"  # noqa: S105  (fake token, test assertion)
    assert kw["server_hostname"] == "example.cloud.databricks.com"  # https:// stripped
    assert "credentials_provider" not in kw


def test_oauth_path_when_no_token(monkeypatch, fake_dbsql, fake_sdk):
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    ldb._connect()
    kw = fake_dbsql.kwargs
    assert fake_sdk["profile"] == "OAUTH"  # default profile
    assert kw["server_hostname"] == "oauth.example.databricks.com"  # host from the profile
    assert callable(kw["credentials_provider"])
    assert "access_token" not in kw


def test_oauth_profile_is_overridable(monkeypatch, fake_dbsql, fake_sdk):
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "PROD")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    ldb._connect()
    assert fake_sdk["profile"] == "PROD"


def test_empty_token_falls_through_to_oauth(monkeypatch, fake_dbsql, fake_sdk):
    # An empty string is not a usable PAT -> OAuth branch, never access_token="".
    monkeypatch.setenv("DATABRICKS_TOKEN", "")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    ldb._connect()
    assert "credentials_provider" in fake_dbsql.kwargs
    assert "access_token" not in fake_dbsql.kwargs
