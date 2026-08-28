"""_fov_registry neutrality: it must NOT import pitch_control / _das / features (ADR-077).

The FOV engine is DEPENDED ON by the pitch-control and DAS layers, so importing them back would
form a cycle. Both the module docstring and ADR-077 assert this neutrality as load-bearing; this gate
enforces it structurally, the way ``tests/gkdv/test_import_allowlist.py`` (ADR-037/043) enforces the
gkdv -> tracking import direction. AST-walk the module's OWN top-level imports -- never a transitive
resolution -- and refuse the three forbidden layers.
"""

from __future__ import annotations

import ast
import pathlib

MODULE = pathlib.Path(__file__).resolve().parents[2] / "silly_kicks" / "tracking" / "_fov_registry.py"

#: The layers _fov_registry must never import -- a cycle if it did, because THEY depend on IT. The
#: allowed neutral imports are _visibility / _kernels / id_compat / _polygon / spadl.config.
FORBIDDEN: tuple[str, ...] = ("pitch_control", "_das", "features")


def _candidate_paths(text: str) -> list[str]:
    """Every dotted module path *text* imports, covering absolute and relative forms.

    For ``from a.b import c`` both ``a.b`` and ``a.b.c`` are emitted (``c`` may be a submodule or a
    symbol -- the tail check below decides), and for a bare relative ``from . import c`` the imported
    NAME ``c`` is emitted (a ``from . import features`` submodule import would otherwise be invisible,
    since ``node.module`` is ``None`` there -- exactly the form the real module uses for ``_kernels``).
    """
    tree = ast.parse(text)
    paths: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = node.module  # None for `from . import X`
            for a in node.names:
                paths.append(f"{base}.{a.name}" if base else a.name)
            if base:
                paths.append(base)
        elif isinstance(node, ast.Import):
            paths.extend(a.name for a in node.names)
    return paths


def _hits(text: str) -> list[str]:
    """Imported module paths whose FINAL component is a forbidden layer."""
    return [p for p in _candidate_paths(text) if p.split(".")[-1] in FORBIDDEN]


def test_fov_registry_does_not_import_forbidden_layers():
    """ADR-077 neutrality: pitch_control / _das / features may depend on _fov_registry, never the
    reverse -- so the engine importing any of them would form a cycle."""
    hits = _hits(MODULE.read_text(encoding="utf-8"))
    assert not hits, (
        f"_fov_registry imports forbidden layer(s) {hits} -- it must import only "
        "_visibility / _kernels / id_compat / _polygon / spadl.config to stay depend-able "
        "without a cycle (module docstring + ADR-077)."
    )


def test_detector_fires_on_planted_forbidden_imports():
    """META: the detector must actually detect -- otherwise the gate silently passes.

    Covers every realistic shape a forbidden import could take (absolute from-import, private
    from-import, submodule from-import, plain import, and the bare-relative ``from . import``)."""
    for planted in (
        "from silly_kicks.tracking import pitch_control\n",
        "from silly_kicks.tracking._das import get_das\n",
        "from silly_kicks.tracking import features\n",
        "from silly_kicks.tracking.pitch_control import PitchControlCache\n",
        "import silly_kicks.tracking.pitch_control\n",
        "from . import features\n",
    ):
        assert _hits(planted), f"detector missed a planted forbidden import: {planted!r}"


def test_detector_does_not_flag_the_allowed_neutral_imports():
    """META: the detector must NOT false-positive on the imports the module actually makes."""
    for allowed in (
        "from silly_kicks.tracking import _kernels\n",
        "from silly_kicks.tracking._visibility import classify_region_observation\n",
        "from silly_kicks.id_compat import canonical_id\n",
        "import silly_kicks.spadl.config as spadlconfig\n",
        "from silly_kicks._polygon import is_convex\n",
        "from . import _kernels\n",
    ):
        assert not _hits(allowed), f"detector false-positives on an allowed import: {allowed!r}"
