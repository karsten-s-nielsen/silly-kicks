"""Every Hub call is bounded, and a `--plan` read has a deadline (HFU spec §13.4, F3).

In retrain launch 2 three `--plan` subprocesses hung 2 h 10 min in `list_bucket_tree` / `get_bucket_paths_info`,
waiting for response headers: huggingface_hub 2.2.0 builds its client with `timeout=None`, and `paginate` passes
`timeout=None` EXPLICITLY, which wins over a client default. A request event hook fills the `None` values.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import textwrap
import threading
import time

import httpx2
import pytest
from huggingface_hub import HfApi, close_session, set_client_factory
from huggingface_hub.utils import _http

from scripts.hf_jobs import campaign as C
from scripts.hf_jobs import driver as D
from scripts.hf_jobs import hub_client as H
from scripts.hf_jobs.store import BucketStore


@pytest.fixture(autouse=True)
def _restore_hub_client():
    """`install_timeouts` replaces the process-wide client factory: restore the default after each test."""
    yield
    set_client_factory(_http.default_client_factory)
    close_session()
    H._INSTALLED = False


def test_hook_fills_none_and_keeps_finite():
    req = httpx2.Request("GET", "http://x/", extensions={"timeout": httpx2.Timeout(None).as_dict()})
    H.bound_request_timeout(req)
    assert req.extensions["timeout"] == {"connect": 10.0, "read": 60.0, "write": 60.0, "pool": 10.0}
    req = httpx2.Request("GET", "http://x/", extensions={"timeout": httpx2.Timeout(5.0).as_dict()})
    H.bound_request_timeout(req)
    assert req.extensions["timeout"]["read"] == 5.0


def test_installed_client_carries_the_hook():
    H.install_timeouts()
    client = _http.get_session()
    assert H.bound_request_timeout in client.event_hooks["request"]
    assert len(client.event_hooks["request"]) >= 2  # the library's own hook is kept


class _SilentServer:
    """Accepts TCP connections and never answers."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.accepted = 0
        self._held: list[socket.socket] = []
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.accepted += 1
            self._held.append(conn)

    def close(self):
        for c in self._held:
            c.close()
        self.sock.close()


def _timeout_in_chain(exc: BaseException | None) -> bool:
    while exc is not None:
        if isinstance(exc, httpx2.TimeoutException):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _call_bounded(fn) -> BaseException | None:
    """Run ``fn`` in a daemon thread; a hang fails the test after 30 s instead of blocking the suite."""
    box: dict[str, BaseException | None] = {}

    def target():
        try:
            fn()
            box["exc"] = None
        except BaseException as exc:
            box["exc"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "the Hub call hung: no timeout reached it"
    return box["exc"]


@pytest.mark.parametrize("call", ["names", "read_bytes"])
def test_bucket_calls_time_out(monkeypatch, call):
    """`names` is the `list_bucket_tree` / `paginate` path (explicit `timeout=None`); `read_bytes` is
    `get_bucket_paths_info`. Both must raise a timeout within the retries."""
    server = _SilentServer()
    try:
        monkeypatch.setitem(H.HUB_TIMEOUT, "read", 0.5)
        monkeypatch.setattr(time, "sleep", lambda s: None)  # no backoff waits (http_backoff calls time.sleep)
        H.install_timeouts()
        store = BucketStore("ns/b", "rk", api=HfApi(endpoint=f"http://127.0.0.1:{server.port}", token=False))
        exc = _call_bounded(lambda: getattr(store, call)("x/y"))
        assert server.accepted >= 1, "the call never reached the local server"
        assert exc is not None and _timeout_in_chain(exc), repr(exc)
    finally:
        server.close()


_HANG = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {repo!r})
    from scripts._stages import arm_plan_deadline
    arm_plan_deadline()
    def _hang():
        time.sleep(60)
    _hang()
    """
)


def test_read_plan_timeout_carries_stack(monkeypatch):
    monkeypatch.setattr(C, "PLAN_TIMEOUT_S", 3)
    code = _HANG.format(repo=str(C.REPO_ROOT))

    def run(argv, **kwargs):  # the real subprocess.run, with the plan argv swapped for a hanging child
        return subprocess.run([sys.executable, "-c", code], **kwargs)  # noqa: S603 -- fixed argv, sys.executable

    inst = C.Instance("job", "public", "default", "m", (), (), "rk")
    with pytest.raises(C.PlanTimeoutError) as info:
        C.read_plan(inst, bucket="ns/b", run=run)
    msg = str(info.value)
    assert "--plan failed (timeout after 3 s)" in msg
    assert 'File "' in msg and "_hang" in msg, msg
    # The dump is the Python-thread dump of the job watch (never the `faulthandler` watchdog, whose dump can
    # end the process it reads): its header starts the detail, and the main thread's hung frame comes first.
    assert "Timeout (" not in msg, msg
    detail = msg.split("(timeout after 3 s): ", 1)[1]
    assert re.match(r"\[alive\] \S+ plan pid=\d+ deadline dump: \d+ thread\(s\)", detail), detail
    assert detail.index(" in _hang") < detail.index(" in <module>")


def test_plan_timeout_is_not_rate_limited():
    assert D._plan_rate_limited(C.PlanTimeoutError("--plan failed (timeout after 600 s): rate limit")) is False


def _timeout(stderr) -> subprocess.TimeoutExpired:
    return subprocess.TimeoutExpired(cmd=["python"], timeout=600, stderr=stderr)


def test_plan_timeout_detail_starts_at_the_dump_header():
    """The detail is the child's dump from its header line; what the child printed before it is dropped."""
    dump = (
        "[alive] - plan pid=7 deadline dump: 2 thread(s), most recent call first\n"
        "Thread MainThread (0x1) [main]:\n"
        '  File "store.py", line 10 in _hang\n'
        '  File "plan.py", line 3 in <module>\n'
        "Thread sk-dump (0x2):\n"
        '  File "liveness.py", line 9 in stack_dump\n'
    )
    err = "a warning line\nanother line\n" + dump
    assert C._plan_timeout_detail(_timeout(err)) == dump
    assert C._plan_timeout_detail(_timeout(err.encode("utf-8"))) == dump  # POSIX gives bytes
    long = dump + "".join(f'  File "deep.py", line {i} in level{i}\n' for i in range(400))
    kept = C._plan_timeout_detail(_timeout("noise\n" + long))
    assert len(kept) == C.PLAN_DUMP_KEEP_CHARS and kept.startswith("[alive] - plan pid=7 deadline dump: ")


def test_plan_timeout_detail_without_a_dump_is_the_stderr_tail():
    err = "x" * 5000 + "the last line"
    got = C._plan_timeout_detail(_timeout(err))
    assert len(got) == C.PLAN_DUMP_KEEP_CHARS and got.endswith("the last line")
    assert C._plan_timeout_detail(_timeout(None)) == ""
