"""Read TOML on every supported interpreter (ADR-128).

``tomllib`` is stdlib from Python 3.11; the CI matrix still runs 3.10, where the API-identical
``tomli`` backport is used (marker-gated in the ``[test]`` extra and resolved in ``uv.lock``). Reading
only: every TOML file in this repo is human-written (the campaign file, ``datasets.toml``, the artifact
manifest). Tool-written records are JSON (decision P2).
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover - exercised on the py3.10 CI leg only
    import tomli as _toml  # pyright: ignore[reportMissingImports]  # py3.10 backport; absent on >=3.11


def load_toml(path) -> dict:
    """Parse the TOML file at ``path``."""
    with open(path, "rb") as fh:
        return _toml.load(fh)


def loads_toml(text: str) -> dict:
    """Parse TOML ``text``."""
    return _toml.loads(text)
