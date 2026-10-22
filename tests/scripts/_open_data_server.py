"""Re-export of the toy open-data server (ADR-128): the implementation lives in ``scripts/_toy_corpora``
because preflight check 5 needs it too; tests keep importing it from here."""

from __future__ import annotations

from scripts._toy_corpora import serve_open_data

__all__ = ["serve_open_data"]
