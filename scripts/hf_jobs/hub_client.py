"""Every huggingface_hub HTTP call is bounded (HFU spec 13.4, F3).

huggingface_hub 2.2.0 builds its shared client with ``timeout=None``, and ``paginate`` (``list_bucket_tree``) passes
``timeout=None`` EXPLICITLY, which ``httpx2`` turns into ``Timeout(None)`` over any client default. In retrain launch
2 three ``--plan`` reads hung 2 h 10 min there, waiting for response headers. A request event hook runs after the
per-request timeout is set and before the send, so it fills each ``None`` value; a finite caller value is kept. A
stalled read then raises ``httpx2.ReadTimeout``, which the library's ``http_backoff`` retries (5 tries).

The same client counts its requests (HFU fit-units spec 17). A second request hook counts every request this
process sends: a retry and a listing page are requests; the bytes of a file come over xet, outside this client,
and are not counted. A ``--plan`` process prints its count in one stderr line; ``campaign.read_plan`` adds it to
the controller's tally, and the driver prints both numbers in its step line. The driver measures its own
margin to the HF rate limit in every launch; no scratch script counts requests."""

from __future__ import annotations

import threading
from typing import Any

#: Seconds per httpx timeout kind. A test lowers ``read``; the hook reads the dict at request time.
HUB_TIMEOUT: dict[str, float] = {"connect": 10.0, "read": 60.0, "write": 60.0, "pool": 10.0}

#: The start of the stderr line with which a ``--plan`` process reports its own request count.
PLAN_REQUESTS_MARK = "[plan] hub requests "

_INSTALLED = False
_COUNT_LOCK = threading.Lock()
_OWN_REQUESTS = 0  # sent by this process
_PLAN_REQUESTS = 0  # sent by the `--plan` processes this process read


def bound_request_timeout(request: Any) -> None:
    """Request event hook: replace each ``None`` value of ``request.extensions["timeout"]`` with ``HUB_TIMEOUT``."""
    current = dict(request.extensions.get("timeout") or {})
    for kind, seconds in HUB_TIMEOUT.items():
        if current.get(kind) is None:
            current[kind] = seconds
    request.extensions = {**request.extensions, "timeout": current}


def count_request(request: Any) -> None:
    """Request event hook: count one HTTP request of this process."""
    global _OWN_REQUESTS
    with _COUNT_LOCK:
        _OWN_REQUESTS += 1


def add_plan_requests(n: int) -> None:
    """Add the request count that one ``--plan`` process reported."""
    global _PLAN_REQUESTS
    with _COUNT_LOCK:
        _PLAN_REQUESTS += n


def request_counts() -> tuple[int, int]:
    """``(own, plans)`` since the process started: the requests this process sent, and the requests that the
    ``--plan`` processes it read reported."""
    with _COUNT_LOCK:
        return _OWN_REQUESTS, _PLAN_REQUESTS


def plan_requests_line() -> str | None:
    """The line a ``--plan`` process prints on stderr when it ends: ``[plan] hub requests <n>``. ``None`` in a
    process without the bounded client (a local store): it made no Hub request."""
    if not _INSTALLED:
        return None
    return f"{PLAN_REQUESTS_MARK}{request_counts()[0]}"


def parse_plan_requests(stderr: str | None) -> int | None:
    """The count of the last `plan_requests_line` in a plan process's stderr; ``None`` when there is none."""
    for line in reversed((stderr or "").splitlines()):
        if line.startswith(PLAN_REQUESTS_MARK):
            text = line[len(PLAN_REQUESTS_MARK) :].strip()
            return int(text) if text.isdigit() else None
    return None


def _bounded_client_factory():
    from huggingface_hub.utils._http import default_client_factory

    client = default_client_factory()
    hooks = client.event_hooks
    requests = [*hooks["request"], bound_request_timeout, count_request]
    client.event_hooks = {"request": requests, "response": hooks["response"]}
    return client


def install_timeouts() -> None:
    """Install the bounded, counting client for every later huggingface_hub call in this process. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    from huggingface_hub import set_client_factory

    set_client_factory(_bounded_client_factory)
    _INSTALLED = True
