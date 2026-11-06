"""Every huggingface_hub HTTP call is bounded (HFU spec 13.4, F3).

huggingface_hub 2.2.0 builds its shared client with ``timeout=None``, and ``paginate`` (``list_bucket_tree``) passes
``timeout=None`` EXPLICITLY, which ``httpx2`` turns into ``Timeout(None)`` over any client default. In retrain launch
2 three ``--plan`` reads hung 2 h 10 min there, waiting for response headers. A request event hook runs after the
per-request timeout is set and before the send, so it fills each ``None`` value; a finite caller value is kept. A
stalled read then raises ``httpx2.ReadTimeout``, which the library's ``http_backoff`` retries (5 tries)."""

from __future__ import annotations

from typing import Any

#: Seconds per httpx timeout kind. A test lowers ``read``; the hook reads the dict at request time.
HUB_TIMEOUT: dict[str, float] = {"connect": 10.0, "read": 60.0, "write": 60.0, "pool": 10.0}

_INSTALLED = False


def bound_request_timeout(request: Any) -> None:
    """Request event hook: replace each ``None`` value of ``request.extensions["timeout"]`` with ``HUB_TIMEOUT``."""
    current = dict(request.extensions.get("timeout") or {})
    for kind, seconds in HUB_TIMEOUT.items():
        if current.get(kind) is None:
            current[kind] = seconds
    request.extensions = {**request.extensions, "timeout": current}


def _bounded_client_factory():
    from huggingface_hub.utils._http import default_client_factory

    client = default_client_factory()
    hooks = client.event_hooks
    client.event_hooks = {"request": [*hooks["request"], bound_request_timeout], "response": hooks["response"]}
    return client


def install_timeouts() -> None:
    """Install the bounded client for every later huggingface_hub call in this process. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    from huggingface_hub import set_client_factory

    set_client_factory(_bounded_client_factory)
    _INSTALLED = True
