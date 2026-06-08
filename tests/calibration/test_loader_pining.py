import pandas as pd
import pytest

import scripts._loader_pining as L


def test_dedupe_gs_frame_records_keeps_first_per_period_frame():
    # Gradient Sports ships some (period, frameNum) records multiple times (up to 16 content-
    # divergent copies). The loader must dedup keep-first so each frame key yields one record;
    # otherwise the linked-frame context fans out (bekkers_pi 3-D crash; inflated PC/DAS inputs).
    frames_json = [
        {"period": 1, "frameNum": 10, "tag": "a"},
        {"period": 1, "frameNum": 10, "tag": "b"},  # duplicate key, divergent content
        {"period": 1, "frameNum": 11, "tag": "c"},
        {"period": 2, "frameNum": 10, "tag": "d"},  # same frameNum, different period -> kept
        {"period": 1, "frameNum": 10, "tag": "e"},  # 3rd copy of (1,10)
    ]
    out = L._dedupe_gs_frame_records(frames_json)
    assert [(f["period"], f["frameNum"]) for f in out] == [(1, 10), (1, 11), (2, 10)]
    assert out[0]["tag"] == "a"  # keep-first


def test_dedupe_gs_frame_records_noop_when_unique():
    frames_json = [{"period": 1, "frameNum": i} for i in range(5)]
    assert L._dedupe_gs_frame_records(frames_json) == frames_json


def test_build_match_with_retry_recovers_from_transient(monkeypatch):
    # A transient download/read blip (e.g. kloppy InputNotFoundError on a partial S3 fetch) must be
    # retried with a fresh temp dir, not crash the whole fold load.
    calls = {"n": 0}

    def _flaky_download(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise OSError("transient S3 blip")
        return {"events": "p"}

    monkeypatch.setattr(L, "_download_artifacts", _flaky_download)
    monkeypatch.setattr(L, "_build_match", lambda *a, **k: ("ACT", "FRM", "H"))
    out = L._build_match_with_retry("skillcorner", "m1", {}, "tok", "url", None, attempts=3, backoff=0)
    assert out == ("ACT", "FRM", "H")
    assert calls["n"] == 2  # failed once, succeeded on the retry


def test_build_match_with_retry_fails_loud_after_attempts(monkeypatch):
    # A genuinely unfetchable match must fail loud after the retries, never silently skip.
    def _always_fail(*a, **k):
        raise OSError("down")

    monkeypatch.setattr(L, "_download_artifacts", _always_fail)
    monkeypatch.setattr(L, "_build_match", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="failed to load after 2 attempts"):
        L._build_match_with_retry("idsse", "M1", {}, "t", "u", None, attempts=2, backoff=0)


def test_two_step_fetch_drops_bearer_on_presigned_get(monkeypatch, tmp_path):
    import urllib.error

    step1 = {}
    step2 = {}

    class _FakeOpener:
        # Step 1 goes through build_opener(...).open — record the bearer + raise the 302.
        def open(self, req, timeout=0):
            step1["auth"] = req.get_header("Authorization")
            step1["url"] = req.full_url
            raise urllib.error.HTTPError(
                req.full_url, 302, "Found", {"Location": "https://s3.example/presigned?sig=x"}, None
            )

    class _FakeResp:
        def __init__(self, body=b"PAYLOAD"):
            self._body = body

        def read(self, *a):
            b, self._body = self._body, b""
            return b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=0):
        # Step 2 (presigned) is called with a BARE URL STRING — no Request, so no bearer header.
        step2["url"] = url
        step2["is_request"] = hasattr(url, "get_header")
        return _FakeResp()

    monkeypatch.setattr(L.urllib.request, "build_opener", lambda *a: _FakeOpener())
    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)

    dest = L._download_to_temp("idsse", "M1", "tracking", "tok", L._DEFAULT_BASE_URL, tmp_path)
    assert dest.read_bytes() == b"PAYLOAD"
    assert step1["auth"] == "Bearer tok"  # step 1 carried the bearer
    assert step2["url"] == "https://s3.example/presigned?sig=x"  # step 2 hit the presigned URL
    assert step2["is_request"] is False  # ...as a bare string, so NO bearer header could attach


def test_load_matches_dispatches_per_provider(monkeypatch, tmp_path):
    # Stub network + conversion; assert orchestration + uniform tuple.
    monkeypatch.setattr(
        L,
        "_list_matches",
        lambda provider, token, base_url: [
            {"id": "M1", "artifacts": {"events": "e", "metadata": "m", "tracking": "t"}}
        ],
    )
    monkeypatch.setattr(L, "_download_to_temp", lambda *a, **k: tmp_path / "artifact.bin")
    built = {
        "actions": pd.DataFrame(
            {
                "game_id": ["M1"],
                "action_id": [0],
                "player_id": [10],
                "period_id": [1],
                "time_seconds": [0.0],
                "team_id": [1],
            }
        ),
        "frames": pd.DataFrame(
            {
                "game_id": ["M1"],
                "period_id": [1],
                "frame_id": [0],
                "player_id": [10],
                "x": [1.0],
                "y": [1.0],
                "team_id": [1],
            }
        ),
        "home": 1,
    }
    monkeypatch.setattr(
        L,
        "_build_match",
        lambda provider, match_id, paths, tracking_limit: (built["actions"], built["frames"], built["home"]),
    )
    rows = list(
        L.load_matches(providers=["idsse"], match_ids={"idsse": ["M1"]}, token="test-token-pining-for-the-data")
    )
    assert len(rows) == 1
    provider, match_id, actions, frames, home = rows[0]
    assert (provider, match_id, home) == ("idsse", "M1", 1)
    assert isinstance(actions, pd.DataFrame) and isinstance(frames, pd.DataFrame)


# --- Extra-time (period 3/4) direction resolution for per-period-absolute providers ---
# (Gradient Sports' convert_to_frames RAISES on ET without home_team_start_left_extratime;
#  the loader must pass it from metadata, or drop ET frames when metadata omits it.)


def test_apply_et_direction_passes_metadata_value_when_present():
    frames = pd.DataFrame({"period_id": [1, 1, 3, 3]})
    out, et = L._apply_et_direction(frames, True, label="gs 123")
    assert et is True
    assert len(out) == 4  # ET frames kept; real direction passed to the converter


def test_apply_et_direction_drops_et_when_metadata_missing():
    frames = pd.DataFrame({"period_id": [1, 1, 3, 4]})
    import pytest

    with pytest.warns(UserWarning, match="ET"):
        out, et = L._apply_et_direction(frames, None, label="gs 123")
    assert et is None
    assert set(out["period_id"]) == {1}  # ET dropped, no guess, no crash


def test_apply_et_direction_noop_when_no_et_periods():
    frames = pd.DataFrame({"period_id": [1, 1, 2, 2]})
    out, et = L._apply_et_direction(frames, None, label="gs 123")
    assert et is None
    assert len(out) == 4  # regular time only: no drop, no warning, param stays None


# --- TF-24 sweep memory bound: load_matches max_per_provider cap (Task 9b) ---


def test_load_matches_max_per_provider_truncates(monkeypatch):
    monkeypatch.setattr(L, "_list_matches", lambda p, t, b: [{"id": str(i), "artifacts": {}} for i in range(5)])
    monkeypatch.setattr(L, "_download_artifacts", lambda *a, **k: {})
    monkeypatch.setattr(L, "_build_match", lambda *a, **k: (None, None, None))
    got = [mid for _p, mid, *_ in L.load_matches(providers=["gradientsports"], max_per_provider=2)]
    assert got == ["0", "1"]


def test_load_matches_no_cap_loads_all(monkeypatch):
    monkeypatch.setattr(L, "_list_matches", lambda p, t, b: [{"id": str(i), "artifacts": {}} for i in range(3)])
    monkeypatch.setattr(L, "_download_artifacts", lambda *a, **k: {})
    monkeypatch.setattr(L, "_build_match", lambda *a, **k: (None, None, None))
    got = [mid for _p, mid, *_ in L.load_matches(providers=["gradientsports"])]
    assert got == ["0", "1", "2"]


def test_build_skillcorner_frames_chains_load_convert_preprocess(monkeypatch):
    # TF-27: the extracted seam must do skillcorner.load -> convert_to_frames -> _preprocess,
    # passing limit + include_empty_frames through. Monkeypatched: no kloppy parse, no network.
    # (function-local imports -> patch the SOURCE module attrs, not L's namespace.)
    sentinel = pd.DataFrame({"x": [1.0, 2.0]})
    seen = {}

    def _fake_load(**kwargs):
        seen["load_kwargs"] = kwargs
        return "DS"

    def _fake_convert(ds):
        seen["ds"] = ds
        return sentinel, None

    monkeypatch.setattr("kloppy.skillcorner.load", _fake_load)
    monkeypatch.setattr("silly_kicks.tracking.kloppy.convert_to_frames", _fake_convert)
    monkeypatch.setattr(L, "_preprocess", lambda f: f.assign(_preprocessed=True))

    out = L.build_skillcorner_frames({"metadata": "m.json", "tracking": "t.jsonl"}, 123)

    assert seen["ds"] == "DS"  # convert_to_frames received the load() result
    assert seen["load_kwargs"]["limit"] == 123
    assert seen["load_kwargs"]["include_empty_frames"] is False
    assert out["_preprocessed"].all() and list(out["x"]) == [1.0, 2.0]  # preprocess applied to convert output
