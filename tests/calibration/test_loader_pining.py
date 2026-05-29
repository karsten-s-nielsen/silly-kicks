import pandas as pd

import scripts._loader_pining as L


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
