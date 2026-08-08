"""The goal-end derivation population is pinned to ONE implementation.

The predicate is SEMANTIC, not node-shaped. An earlier draft of the design shipped an
``IfExp``-only rule built from the shapes its author had already found -- every one a
ternary -- so executing it confirmed those and nothing else. Four ``if``/``else``
statements were structurally invisible to it. A detector derived from its own sample is
self-confirming, which is why the non-vacuity witness below is written in the spelling
that was missing.

Deliberately NOT matched: dict literals. The only two matches in the tree were a
``grid_spec`` metadata dict and ``{1: 0, 2: 45, 3: 90, 4: 105, 5: 120}`` -- period-start
MINUTES, where the 105 is not metres.
"""

from __future__ import annotations

import ast
import pathlib
import re

PITCHY = re.compile(r"(FIELD_LENGTH|PITCH_LENGTH)", re.I)

# SB_FIELD_LENGTH is 120.0 (StatsBomb native), NOT a pitch end. Without this exclusion a
# `0.0 / SB_FIELD_LENGTH` binding is a false positive -- and the "fix" would then be an
# exemption for something that is not the defect.
_NOT_PITCH = frozenset({"SB_FIELD_LENGTH", "SB_FIELD_WIDTH"})

ROOT = pathlib.Path(__file__).resolve().parents[2] / "silly_kicks"

SEAM = "silly_kicks/tracking/_gk_resolve.py"

_EXEMPT: dict[str, str] = {
    "silly_kicks/tracking/_shot_goalmouth.py": (
        "The documented PSO / degenerate ball-mean fallback: a LAST-RESORT end for when the goal "
        "map itself is degenerate, so by construction it cannot consult the map."
    ),
}


def _is_zero(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and float(node.value) == 0.0


def _is_pitch(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and float(node.value) == 105.0:
        return True
    name = getattr(node, "id", None) or getattr(node, "attr", None)
    if not isinstance(name, str) or name in _NOT_PITCH:
        return False
    return bool(PITCHY.search(name))


def _pair(left: ast.AST, right: ast.AST) -> bool:
    return (_is_zero(left) and _is_pitch(right)) or (_is_pitch(left) and _is_zero(right))


def _assigned(stmts: list[ast.stmt]) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for stmt in stmts:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            out[stmt.targets[0].id] = stmt.value
    return out


def goal_end_sites(tree: ast.AST) -> list[int]:
    """Lines binding ONE name to ``{0.0, pitch-length}``, through any construct."""
    hits: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp) and _pair(node.body, node.orelse):
            hits.append(node.lineno)
        elif isinstance(node, ast.If) and node.orelse:
            body, orelse = _assigned(node.body), _assigned(node.orelse)
            for name in set(body) & set(orelse):
                if _pair(body[name], orelse[name]):
                    hits.append(node.lineno)
        elif isinstance(node, ast.Call):
            fn = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if fn == "where" and len(node.args) == 3 and _pair(node.args[1], node.args[2]):
                hits.append(node.lineno)
    return sorted(set(hits))


def _scan() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT.parent).as_posix()
        sites = goal_end_sites(ast.parse(path.read_text(encoding="utf-8")))
        if sites:
            found[rel] = sites
    return found


def test_goal_end_derivation_lives_in_exactly_one_place() -> None:
    found = _scan()
    assert found, "the scanner found nothing -- it is broken, not the tree clean"
    assert SEAM in found, "the seam itself derives no goal end -- this gate would pass vacuously"
    assert len(found[SEAM]) == 1, f"the seam must derive it ONCE, found {found[SEAM]}"
    extra = {k: v for k, v in found.items() if k not in ({SEAM} | set(_EXEMPT))}
    assert not extra, (
        "goal-end derivation outside the pinned seam: "
        + "; ".join(f"{k}:{v}" for k, v in sorted(extra.items()))
        + "\nCall silly_kicks.tracking.resolve_defended_goals instead."
    )


def test_every_exempt_entry_actually_matches() -> None:
    """ADR-051 both-directions: an exemption that can never match is a lie, so it fails too."""
    stale = sorted(set(_EXEMPT) - set(_scan()))
    assert not stale, f"_EXEMPT names files with no goal-end site: {stale}"


def test_the_predicate_catches_an_IF_STATEMENT_fork() -> None:
    """Non-vacuity, in the spelling an IfExp-only predicate was BLIND to.

    A ternary plant would only prove the gate catches copy-paste of the shapes the predicate
    was built from, which is the failure this predicate exists to avoid repeating.
    """
    planted = ast.parse(
        "def sneaky(team, home):\n"
        "    if team == home:\n"
        "        end = 0.0\n"
        "    else:\n"
        "        end = 105.0\n"
        "    return end\n"
    )
    assert goal_end_sites(planted), "an if/else fork must be detected"


def test_the_predicate_catches_ternary_and_np_where_forks() -> None:
    for src in (
        "def a(c):\n    return 0.0 if c else 105.0\n",
        "import numpy as np\ndef b(c):\n    return np.where(c, 0.0, 105.0)\n",
    ):
        assert goal_end_sites(ast.parse(src)), f"undetected fork:\n{src}"


def test_the_predicate_ignores_a_period_start_MINUTES_dict() -> None:
    """The 105 in period-start minutes is not metres."""
    assert not goal_end_sites(ast.parse("def t():\n    return {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}\n"))


def test_the_predicate_ignores_the_STATSBOMB_pitch_length() -> None:
    """SB_FIELD_LENGTH is 120.0, a different pitch convention, not a goal end."""
    src = "SB_FIELD_LENGTH = 120.0\ndef t(c):\n    return 0.0 if c else SB_FIELD_LENGTH\n"
    assert not goal_end_sites(ast.parse(src))
