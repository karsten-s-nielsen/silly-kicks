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
    monkeypatch.delenv("DATABRICKS_AUTH", raising=False)
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapiXXX")
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    ldb._connect()
    kw = fake_dbsql.kwargs
    assert kw["access_token"] == "dapiXXX"  # noqa: S105  (fake token, test assertion)
    assert kw["server_hostname"] == "example.cloud.databricks.com"  # https:// stripped
    assert "credentials_provider" not in kw


def test_oauth_path_when_no_token(monkeypatch, fake_dbsql, fake_sdk):
    monkeypatch.delenv("DATABRICKS_AUTH", raising=False)
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
    monkeypatch.delenv("DATABRICKS_AUTH", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "PROD")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    ldb._connect()
    assert fake_sdk["profile"] == "PROD"


def test_empty_token_falls_through_to_oauth(monkeypatch, fake_dbsql, fake_sdk):
    monkeypatch.delenv("DATABRICKS_AUTH", raising=False)
    # An empty string is not a usable PAT -> OAuth branch, never access_token="".
    monkeypatch.setenv("DATABRICKS_TOKEN", "")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    ldb._connect()
    assert "credentials_provider" in fake_dbsql.kwargs
    assert "access_token" not in fake_dbsql.kwargs


def test_DATABRICKS_AUTH_oauth_overrides_a_present_token(monkeypatch, fake_dbsql, fake_sdk):
    """The whole point: a stale PAT sitting in the environment must not pre-empt working OAuth.

    Measured on the maintainer's machine: `DATABRICKS_TOKEN` held a dead 36-char `dapi` PAT with
    `DATABRICKS_CONFIG_PROFILE` unset, so every Databricks-backed driver took the PAT branch and
    failed with an error naming the WORKSPACE -- never the environment variable that chose it.
    """
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-stale")
    monkeypatch.setenv("DATABRICKS_AUTH", "oauth")
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    ldb._connect()
    assert fake_dbsql.kwargs is not None
    assert fake_dbsql.kwargs.get("credentials_provider") is not None
    assert "access_token" not in fake_dbsql.kwargs


def test_DATABRICKS_AUTH_pat_without_a_token_RAISES(monkeypatch, fake_dbsql, fake_sdk):
    """Explicitly asking for PAT with no PAT present is an error, not a silent fallback to OAuth --
    a silent fallback would make the flag look honoured when it was ignored."""
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("DATABRICKS_AUTH", "pat")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    with pytest.raises(RuntimeError, match="DATABRICKS_AUTH=pat"):
        ldb._connect()


def test_an_UNRECOGNISED_DATABRICKS_AUTH_raises_rather_than_being_ignored(monkeypatch, fake_dbsql, fake_sdk):
    """A typo (`DATABRICKS_AUTH=OAUTH2`, `=token`) must not silently fall through to the historic
    precedence -- that is the failure this variable exists to end, reintroduced one typo lower."""
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-stale")
    monkeypatch.setenv("DATABRICKS_AUTH", "oauth2")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    with pytest.raises(RuntimeError, match="must be 'pat', 'oauth' or unset"):
        ldb._connect()


def test_an_UNSET_DATABRICKS_AUTH_preserves_the_historic_precedence(monkeypatch, fake_dbsql, fake_sdk):
    """The compatibility half of the band. CI and every legacy setup leave the variable unset, and
    for them a non-empty token must still win exactly as before -- otherwise this "override" is a
    silent behaviour change for every existing caller rather than an opt-in."""
    monkeypatch.delenv("DATABRICKS_AUTH", raising=False)
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapiXXX")
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    ldb._connect()
    assert fake_dbsql.kwargs["access_token"] == "dapiXXX"  # noqa: S105  (fake token, assertion)
