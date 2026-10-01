"""Fail-loud read-only memo-read seam for ``validate_construct_validity`` (Phase 2 MCP).

Surfaces a metric family's RECORDED construct-validity verdict from its committed
``docs/research/<family>/…json`` memo. It NEVER re-runs a validate driver, loads a corpus, or
touches the tree. It RAISES (never returns a verdict) on a memo that is absent, unreadable, empty,
produced from a dirty tree (``run_tree_dirty: true``), or missing provenance (no ``run_commit`` /
``run_tree_dirty`` keys — the current ``xtgk_possession_value/gate.json`` case). Regenerating that
memo with provenance is a deferred owner-corpus run (spec §11).
"""

from __future__ import annotations

import json
from pathlib import Path

#: family -> memo path, relative to the research root. Heterogeneous by design: the two
#: ``*_construct_validity`` memos carry ``run_commit``/``run_tree_dirty``/``verdicts``; the xtgk
#: gate.json does not (→ fail-loud until regenerated, spec §4b/§11).
_FAMILY_MEMOS: dict[str, str] = {
    "gk_decision": "gk_decision_construct_validity/metrics.json",
    "territorial_defense": "territorial_defense_construct_validity/metrics.json",
    "xtgk_possession_value": "xtgk_possession_value/gate.json",
}

_DEFAULT_RESEARCH_ROOT = Path(__file__).resolve().parents[2] / "docs" / "research"


def families() -> tuple[str, ...]:
    """The metric families the tool dispatches on (all three; xtgk currently fails loud)."""
    return tuple(_FAMILY_MEMOS)


def memo_path_for(metric_family: str, *, research_root: Path | str | None = None) -> Path:
    if metric_family not in _FAMILY_MEMOS:
        raise ValueError(f"unknown metric_family {metric_family!r}; known: {sorted(_FAMILY_MEMOS)}")
    root = Path(research_root) if research_root is not None else _DEFAULT_RESEARCH_ROOT
    return root / _FAMILY_MEMOS[metric_family]


def read_validity_memo(metric_family: str, *, research_root: Path | str | None = None) -> dict:
    """Read + validate a family's construct-validity memo. RAISES on any untrustworthy state.

    A missing ``run_commit``/``run_tree_dirty`` is an explicit raise (not a ``KeyError``): the xtgk
    ``gate.json`` records neither, so the tool refuses rather than surfacing an unprovenanced verdict.
    """
    memo_path = memo_path_for(metric_family, research_root=research_root)
    if not memo_path.is_file():
        raise FileNotFoundError(f"validity memo absent: {memo_path} — run the {metric_family} validate_* driver first")
    try:
        memo = json.loads(memo_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"validity memo unreadable ({memo_path}): {exc}") from exc
    if not isinstance(memo, dict) or not memo:
        raise ValueError(f"validity memo empty or not an object: {memo_path}")
    if "run_commit" not in memo or "run_tree_dirty" not in memo:
        raise ValueError(
            f"validity memo {memo_path} lacks provenance (run_commit/run_tree_dirty); "
            f"regenerate the {metric_family} memo with provenance before trusting its verdict"
        )
    if memo["run_tree_dirty"] is True:
        raise ValueError(
            f"validity memo {memo_path} was produced from a DIRTY tree (run_tree_dirty=true); "
            "its verdict is untrustworthy"
        )
    if "verdicts" not in memo:
        raise ValueError(f"validity memo {memo_path} has provenance but no 'verdicts' block")
    return memo


def recorded_verdict(memo: dict) -> dict:
    """The recorded verdict block (the analyst-facing construct-validity result)."""
    return memo["verdicts"]
