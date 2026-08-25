import pandas as pd

import scripts._loader_databricks as L
import silly_kicks.spadl.config as spadlconfig


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
    # ADR-068 batched the per-match WHERE into ONE `match_id IN (...)` query per table, but each id
    # is STILL a generated bound placeholder (%(mid0)s, ...) -- the injection-safety property holds.
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
        # IN-list of generated bound placeholders; the malicious id is bound, never in the SQL text.
        assert "match_id IN (%(mid0)s)" in sql and "m1; DROP TABLE x" not in sql
        assert params == {"mid0": "m1; DROP TABLE x"}


def _mart_row(match_id, action_type, action_result, **kw):
    base = dict(
        match_id=match_id,
        start_x=10.0,
        start_y=20.0,
        end_x=30.0,
        end_y=40.0,
        action_type=action_type,
        action_result=action_result,
    )
    base.update(kw)
    return base


def test_shape_action_values_maps_strings_to_int_codes():
    df = pd.DataFrame([_mart_row(101, "pass", "success"), _mart_row(101, "dribble", "fail")])
    out = L.shape_action_values(df)
    assert out.loc[0, "type_id"] == spadlconfig.actiontype_id["pass"]
    assert out.loc[0, "result_id"] == spadlconfig.result_id["success"]
    assert out.loc[1, "type_id"] == spadlconfig.actiontype_id["dribble"]
    assert out.loc[1, "result_id"] == spadlconfig.result_id["fail"]


def test_shape_action_values_uses_nullable_int_dtype():
    out = L.shape_action_values(pd.DataFrame([_mart_row(1, "pass", "success")]))
    assert str(out["type_id"].dtype) == "Int64"
    assert str(out["result_id"].dtype) == "Int64"


def test_shape_action_values_aliases_match_id_to_game_id():
    out = L.shape_action_values(pd.DataFrame([_mart_row(777, "pass", "success")]))
    assert (out["game_id"] == 777).all()


def test_shape_action_values_tolerates_unmapped_vocab():
    # Unknown action_type/result -> <NA> (the move filter drops it; must not raise).
    df = pd.DataFrame([_mart_row(1, "teleport", "success"), _mart_row(1, "pass", "quantum")])
    out = L.shape_action_values(df)
    assert pd.isna(out.loc[0, "type_id"])
    assert pd.isna(out.loc[1, "result_id"])


def test_shape_action_values_preserves_coordinates():
    out = L.shape_action_values(pd.DataFrame([_mart_row(1, "pass", "success", start_x=5.5, end_y=63.0)]))
    assert out.loc[0, "start_x"] == 5.5
    assert out.loc[0, "end_y"] == 63.0
