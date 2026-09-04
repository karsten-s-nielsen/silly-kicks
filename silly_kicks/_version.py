"""Single source of truth for the package version (ADR-079).

This is the ONLY place the version is written. ``pyproject.toml`` declares ``dynamic = ["version"]``
and hatchling reads the literal from here (``[tool.hatch.version] path``); ``silly_kicks/__init__.py``
re-exports it. Bumping this one line is the whole source change for a release -- ``uv.lock`` follows
from ``uv lock``, and the wheel/sdist metadata follow from hatchling. Kept import-free so tooling can
read the version without importing the (heavy) ``silly_kicks`` package.
"""

from __future__ import annotations

__version__ = "4.108.0"
