"""Declared input contracts for research-artifact drivers (ADR-054).

A driver declares WHICH SYMBOLS its numbers depend on, never what those symbols currently contain.
When ``SHOT_ARM_CONFOUNDERS`` gains a column or ``GEOMETRY_VERSION`` bumps, the digest moves without
anyone editing the driver. That is the difference between this and "a human writes a list": the
residual failure mode is "forgot to reference a symbol at all" -- narrow and visible -- rather than
"typed a list that later went stale", which is what a literal declaration would ship.

Deliberately the same shape as ADR-050's ``feature_contract``, because that pattern is already
built, reviewed and trusted here.

KNOWN LIMIT, declared rather than discovered: this catches code drift, not under-declaration. A
driver that never references ``theta`` digests stably forever. Two alternatives were considered and
rejected in the spec (S1.4): deriving the declaration from imports re-creates the caller-sweep blind
spot IN CODE, and a runtime column-coverage check would make the cycle's central mechanism depend on
new per-shard instrumentation rather than a proven pattern. The stated trigger for revisiting is an
invalidation that PR 6 or PR 7 turns up and this mechanism missed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

CONTRACT_VERSION = 1


def _canonical(value: Any) -> Any:
    """Reduce ``value`` to a JSON-serializable form with a deterministic ordering.

    Sets are sorted by ``repr`` so iteration order cannot move the digest. Dict KEYS are sorted the
    same way rather than naturally: ``sorted({1: "a", "b": "c"})`` raises
    ``TypeError: '<' not supported between instances of 'str' and 'int'``, and ``covariates`` is
    caller-supplied, so that crash would land at driver-run time rather than at gate time.
    """
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=repr)}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_canonical(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def contract_digest(parts: dict) -> str:
    """SHA256 over the canonical JSON of everything except the digest itself."""
    body = {k: _canonical(v) for k, v in sorted(parts.items()) if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def declare_inputs(**parts: Any) -> dict:
    """Build a driver's input contract, written into its artifact beside ``run_commit``.

    Every caller must pass ``driver="<script stem>"`` so a committed artifact can be keyed back to
    the driver that produced it.
    """
    out: dict[str, Any] = {"version": CONTRACT_VERSION}
    out.update({k: _canonical(v) for k, v in parts.items()})
    out["digest"] = contract_digest(out)
    return out
