import pandas as pd

import scripts._loader_databricks as L


class _FakeCursor:
    def __init__(self, frames_df, events_df):
        self._frames, self._events = frames_df, events_df
        self._last = None
        self._drained = False

    def execute(self, sql, params=None):  # M5: now (sql, params)
        self._last = "tracking" if "_tracking" in sql else "events" if "_events" in sql else "ids"
        self._drained = False

    def _rows(self):
        if self._last == "ids":
            return [("m1",)]
        df = self._frames if self._last == "tracking" else self._events
        return list(df.itertuples(index=False, name=None))

    def fetchmany(self, n):  # L4: batched fetch — return all once, then empty
        if self._drained:
            return []
        self._drained = True
        return self._rows()

    @property
    def description(self):
        df = self._frames if self._last == "tracking" else self._events
        cols = ["match_id"] if self._last == "ids" else list(df.columns)
        return [(c,) for c in cols]

    def close(self):
        pass


class _FakeConn:
    def __init__(self, f, e):
        self._f, self._e = f, e

    def cursor(self):
        return _FakeCursor(self._f, self._e)

    def close(self):
        pass


def test_databricks_loader_provider_allowlist():
    # M5: a non-allowlisted provider must be rejected before any SQL is built.
    import pytest

    with pytest.raises(ValueError, match="not in allowlist"):
        L._table("'; DROP TABLE x; --", "tracking")


def test_databricks_loader_parameterizes_match_id(monkeypatch):
    # The match_id must be passed as a bound parameter, never interpolated into the SQL string.
    seen = []

    class _SpyCursor(_FakeCursor):
        def execute(self, sql, params=None):
            seen.append((sql, params))
            super().execute(sql, params)

    frames_df = pd.DataFrame(
        {
            "match_id": ["m1"],
            "period_id": [1],
            "frame_id": [0],
            "player_id": [10],
            "x": [10.0],
            "y": [10.0],
            "team_id": [1],
        }
    )
    events_df = pd.DataFrame({"match_id": ["m1"], "action_id": [0], "team_id": [1]})

    class _SpyConn(_FakeConn):
        def cursor(self):
            return _SpyCursor(self._f, self._e)

    monkeypatch.setattr(L, "_connect", lambda: _SpyConn(frames_df, events_df))
    # sportec _convert raises NotImplemented for a non-sportec provider with no mapping; use the
    # parameterization spy on the WHERE queries only (stop before _convert by catching).
    gen = L.load_matches(providers=["sportec"], match_ids={"sportec": ["m1; DROP TABLE x"]})
    try:
        next(gen)
    except Exception:  # _convert may raise on the fake frames — we only assert SQL safety here
        pass
    where_calls = [c for c in seen if "WHERE match_id" in c[0]]
    assert where_calls, "expected a parameterized WHERE query"
    for sql, params in where_calls:
        assert "%(mid)s" in sql and "m1; DROP TABLE x" not in sql  # bound, not interpolated
        assert params == {"mid": "m1; DROP TABLE x"}
