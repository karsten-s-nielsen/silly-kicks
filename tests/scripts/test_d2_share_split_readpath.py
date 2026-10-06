"""Anti-rot for the per-variant baseline share split (ADR-112 follow-up, option C).

The baseline combined table is stored per variant (``level_combined_path(out, BASELINE_LEVEL, variant)``). A reader
that calls ``level_combined_path(out, BASELINE_LEVEL)`` WITHOUT a variant reads a file that no longer exists -> a
silent empty read. This gate derives every ``level_combined_path`` call in ``calibrate_coordination.py`` by AST and
asserts no call site whose level is the LITERAL ``BASELINE_LEVEL`` omits the variant (teeth proven by a planted
snippet). Pure AST -- no ruthless import.

Scope (DLA-IMPL-01): this is a LITERAL-``BASELINE_LEVEL`` static check. ``_scored_pairs`` reads with a dynamic ``k``
(``level_combined_path(out, k, "base") if k == BASELINE_LEVEL else level_combined_path(out, k)``) -- correct (it reads
the baseline "base" variant file), but not a literal, so it is out of this gate's reach by construction. Its
correctness is backstopped by the D-5 whole-``calibration.json`` golden
(test_d2_layer_a_share_split_identity.test_whole_calibration_json_byte_identical_to_golden), which asserts the WHOLE
artifact incl ``population`` -- the field ``_scored_pairs`` feeds -- so a wrong baseline read there changes the golden.
"""

from __future__ import annotations

import ast
import pathlib
import sys

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_CALIBRATE = _SCRIPTS / "calibrate_coordination.py"


def _stacked_baseline_reads(src: str) -> list[int]:
    """Line numbers of every ``level_combined_path(..., BASELINE_LEVEL)`` call that omits the variant (positional 3rd
    arg or ``variant=`` kwarg) -- i.e. reads the non-existent stacked baseline file."""
    bad: list[int] = []
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "level_combined_path"):
            continue
        level = node.args[1] if len(node.args) >= 2 else None
        is_baseline_literal = isinstance(level, ast.Name) and level.id == "BASELINE_LEVEL"
        has_variant = len(node.args) >= 3 or any(kw.arg == "variant" for kw in node.keywords)
        if is_baseline_literal and not has_variant:
            bad.append(node.lineno)
    return bad


def test_no_baseline_reader_reads_a_stacked_file():
    bad = _stacked_baseline_reads(_CALIBRATE.read_text(encoding="utf-8"))
    assert not bad, (
        f"level_combined_path(..., BASELINE_LEVEL) without a variant (reads a non-existent stacked file): {bad}"
    )


def test_the_gate_has_teeth():
    # a planted stacked baseline read IS flagged; a per-variant read is NOT.
    planted = "x = level_combined_path(out, BASELINE_LEVEL)\n"
    assert _stacked_baseline_reads(planted) == [1]
    ok = (
        "a = level_combined_path(out, BASELINE_LEVEL, 'base')\n"
        "b = level_combined_path(out, BASELINE_LEVEL, variant='v')\n"
        "c = level_combined_path(out, some_prep_level)\n"
    )
    assert _stacked_baseline_reads(ok) == []
