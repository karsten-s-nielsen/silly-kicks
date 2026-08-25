"""Task 18 (ADR-068): the databricks loader batches ONE IN-list query per table instead of two
queries per match (2N round-trips), splitting client-side dtype-safely."""

import pandas as pd

import scripts._loader_databricks as ld


class _FakeCursor:
    def close(self):
        pass


class _FakeConn:
    def cursor(self):
        return _FakeCursor()

    def close(self):
        pass


def test_batches_two_queries_and_splits_per_match(monkeypatch):
    all_frames = pd.DataFrame({"match_id": [1, 1, 2], "frame_id": [0, 1, 0], "x": [1.0, 2.0, 3.0]})
    all_events = pd.DataFrame({"match_id": [1, 2, 2], "ev": [10, 20, 21]})
    seen: list[str] = []

    def _fake_query(cur, sql, params=None):
        seen.append(sql)
        return all_frames.copy() if "T_TRACK" in sql else all_events.copy()

    monkeypatch.setattr(ld, "_connect", lambda: _FakeConn())
    monkeypatch.setattr(ld, "_table", lambda provider, kind: {"tracking": "T_TRACK", "events": "T_EVT"}[kind])
    monkeypatch.setattr(ld, "_query_param", _fake_query)
    monkeypatch.setattr(ld, "_convert", lambda provider, raw_events, raw_frames: (raw_events, raw_frames, "home"))

    # str lookup ids vs the int `match_id` column above -> exercises the dtype-safe split.
    out = list(ld.load_matches(providers=["skillcorner"], match_ids={"skillcorner": ["1", "2"]}, tracking_limit=None))

    # TWO data queries (one IN-list per table), NOT 2 per match (== 4).
    assert len(seen) == 2
    assert all("IN (" in s for s in seen)
    # Correct per-match split, in ids order.
    assert [mid for _, mid, *_ in out] == ["1", "2"]
    (_p1, _m1, ev1, fr1, _h1), (_p2, _m2, ev2, fr2, _h2) = out
    assert fr1["match_id"].tolist() == [1, 1] and fr2["match_id"].tolist() == [2]
    assert ev1["ev"].tolist() == [10] and ev2["ev"].tolist() == [20, 21]
