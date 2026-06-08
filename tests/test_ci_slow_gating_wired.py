"""Structural guard: the CI matrix partitions into exactly one bulk pytest process per leg.

Prevents the cardinal silently-skipping sin -- if the primary-leg predicate drifts (or a bulk step
is dropped), a leg can run zero bulk tests yet go green, and the slow tests run nowhere. We assert
the SEMANTIC partition on the BULK steps -- which step activates per resolved leg and its -m gating
-- not mere string presence (three steps carry matrix.primary expressions; presence proves nothing).

See ADR-023.
"""

from __future__ import annotations

import pathlib

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CI = _REPO / ".github" / "workflows" / "ci.yml"


def _guard(expr: object) -> str:
    """Normalize a step ``if:`` to its inner expression: whitespace-stripped, ``${{ }}`` unwrapped
    (GitHub accepts both ``${{ matrix.primary }}`` and brace-less ``matrix.primary``)."""
    s = "".join(str(expr).split())
    if s.startswith("${{") and s.endswith("}}"):
        s = s[3:-2]
    return s


def test_ci_bulk_steps_partition_with_slow_gating() -> None:
    ci = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    test_job = ci["jobs"]["test"]

    # exactly one primary leg, from a single source of truth (the matrix include flag)
    include = test_job["strategy"]["matrix"].get("include", [])
    primaries = [e for e in include if e.get("primary") is True]
    assert len(primaries) == 1, f"expected exactly one matrix include with primary: true, got {primaries}"

    # the two BULK (non-benchmark) pytest steps must partition the matrix on matrix.primary
    bulk = [
        s for s in test_job["steps"] if "run" in s and "pytest tests/" in s["run"] and "--benchmark-skip" in s["run"]
    ]
    assert len(bulk) == 2, f"expected exactly two bulk steps, got {len(bulk)}: {[s.get('run') for s in bulk]}"
    guards = {_guard(s.get("if", "")): s for s in bulk}
    assert set(guards) == {"matrix.primary", "!matrix.primary"}, f"bulk steps not complementary: {set(guards)}"

    # the gating EFFECT: non-primary excludes slow; primary runs everything (incl slow)
    assert "not slow" in guards["!matrix.primary"]["run"], "non-primary bulk step must exclude slow"
    assert "not slow" not in guards["matrix.primary"]["run"], "primary bulk step must run slow (no 'not slow')"


def test_slow_marker_set_is_non_empty() -> None:
    hits = sum(1 for p in (_REPO / "tests").rglob("*.py") if "pytest.mark.slow" in p.read_text(encoding="utf-8"))
    assert hits >= 1, "no tests carry @pytest.mark.slow; the gating would be a no-op"
