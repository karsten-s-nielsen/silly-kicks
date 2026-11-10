"""optuna is pinned to 5.0.0 (HFU spec 14.10a, owner 2026-11-06).

Seven of the eight committed TPE-tuned xshot/xcross models were tuned under optuna 5.0.0, and their best
parameters reproduce only under it (spec 14.10 I1): an unpinned ``ruthless-efficiency[optuna]`` resolved 4.8.0
in launch 3 and reproduced none. Every extra that pulls the Optuna substrate therefore pins the floor, and the
lock resolves exactly 5.0.0.
"""

from __future__ import annotations

from pathlib import Path

from scripts._toml import load_toml

ROOT = Path(__file__).resolve().parents[1]


def test_optuna_floor_and_lock():
    extras = load_toml(ROOT / "pyproject.toml")["project"]["optional-dependencies"]
    with_optuna = {k for k, deps in extras.items() if any(d.startswith("ruthless-efficiency[optuna]") for d in deps)}
    assert with_optuna == {"calibration", "test", "train"}
    for name in sorted(with_optuna):
        assert "optuna>=5.0.0,<6" in extras[name], name
    lock = load_toml(ROOT / "uv.lock")
    assert [p["version"] for p in lock["package"] if p["name"] == "optuna"] == ["5.0.0"]
