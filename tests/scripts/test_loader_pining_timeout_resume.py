"""F6 (TF-46 Arm 1 Part 2, spec section 14.3): the pining loader RESUMES a long presigned read after a
mid-stream timeout -- a transient ``TimeoutError`` re-opens the URL with a ``Range`` request for the bytes it
does not hold yet and appends them, rather than failing the whole corpus pass or re-reading from byte 0. A server
that ignores the range (200) restarts the file. A non-timeout error is not retried."""

from __future__ import annotations

import email.message
import urllib.error
import urllib.request

import pytest

from scripts import _loader_pining


class _FakeResp:
    """A minimal streaming response context manager: the given chunks, then EOF. ``fail_after`` raises a
    ``TimeoutError`` once that many chunks were served (a mid-stream stall)."""

    def __init__(self, chunks=(b"payload-bytes",), *, status=200, fail_after=None) -> None:
        self._chunks = [*chunks, b""]
        self.status = status
        self._served = 0
        self._fail_after = fail_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, _n):
        if self._fail_after is not None and self._served >= self._fail_after:
            raise TimeoutError("read timed out")
        self._served += 1
        return self._chunks.pop(0)


class _FakeOpener:
    """``opener.open`` always 302-redirects to a presigned URL (the loader's two-step contract)."""

    def open(self, _req, timeout=60):  # mirrors urllib's opener.open name
        hdrs = email.message.Message()
        hdrs["Location"] = "http://presigned/object"
        raise urllib.error.HTTPError("http://api/redirect", 302, "found", hdrs, None)


def test_download_resumes_after_one_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a, **_k: _FakeOpener())

    calls = {"n": 0}

    def _fake_urlopen(_location, timeout=600):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("read timed out")  # first presigned read stalls
        return _FakeResp()  # the resume succeeds

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    dest = _loader_pining._download_to_temp(
        "skillcorner", "123", "tracking.jsonl", token="t", base_url="http://api", dest_dir=tmp_path
    )
    assert dest.exists()
    assert dest.read_bytes() == b"payload-bytes"
    assert calls["n"] == 2  # one timeout, one successful resume


def test_download_gives_up_after_the_retry_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a, **_k: _FakeOpener())

    def _always_timeout(_location, timeout=600):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(urllib.request, "urlopen", _always_timeout)

    with pytest.raises(RuntimeError, match="timed out"):
        _loader_pining._download_to_temp(
            "skillcorner", "123", "tracking.jsonl", token="t", base_url="http://api", dest_dir=tmp_path
        )


def _range_of(req) -> str | None:
    return req.get_header("Range") if hasattr(req, "get_header") else None


def test_mid_stream_timeout_resumes_with_a_range_request_and_appends(tmp_path, monkeypatch):
    # The first read serves 6 bytes then stalls; the resume asks for bytes=6- and gets 206 with the rest.
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a, **_k: _FakeOpener())
    seen: list[str | None] = []
    headers_seen: list[dict] = []

    def _fake_urlopen(req, timeout=600):
        seen.append(_range_of(req))
        headers_seen.append(dict(req.header_items()) if hasattr(req, "header_items") else {})
        if len(seen) == 1:
            assert isinstance(req, str)  # a fresh presigned read is a BARE URL (no header can attach)
            return _FakeResp((b"payloa", b"d-bytes"), fail_after=1)
        return _FakeResp((b"d-bytes",), status=206)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    dest = _loader_pining._download_to_temp(
        "skillcorner", "123", "tracking.jsonl", token="t", base_url="http://api", dest_dir=tmp_path
    )
    assert seen == [None, "bytes=6-"]
    assert dest.read_bytes() == b"payload-bytes"
    assert all(set(h) <= {"Range"} for h in headers_seen)  # a resume carries the byte Range ONLY, never a bearer


def test_a_server_ignoring_the_range_restarts_the_file(tmp_path, monkeypatch):
    # 200 to a range request = the whole body again: the partial is rewritten, never double-appended.
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a, **_k: _FakeOpener())
    seen: list[str | None] = []

    def _fake_urlopen(req, timeout=600):
        seen.append(_range_of(req))
        if len(seen) == 1:
            return _FakeResp((b"payloa", b"d-bytes"), fail_after=1)
        return _FakeResp((b"payload-bytes",), status=200)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    dest = _loader_pining._download_to_temp(
        "skillcorner", "123", "tracking.jsonl", token="t", base_url="http://api", dest_dir=tmp_path
    )
    assert seen == [None, "bytes=6-"]
    assert dest.read_bytes() == b"payload-bytes"
