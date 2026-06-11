import importlib
import subprocess
import sys


def test_top_level_import_does_not_pull_calibration_or_heavy_deps():
    # Fresh subprocess: `import silly_kicks` must NOT import the optional subpackage or its heavy
    # deps (ruthless/xgboost), so `import silly_kicks` stays dependency-light (L1 — a real,
    # falsifiable guard, not `assert ... or True`).
    code = (
        "import sys; import silly_kicks; "
        "bad=[m for m in ('silly_kicks.calibration','ruthless','xgboost') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)  # noqa: S603
    assert proc.returncode == 0, f"top-level import leaked: {proc.stdout.strip()}"


def test_top_level_init_has_no_calibration_import():
    src = importlib.import_module("silly_kicks").__file__
    with open(src, encoding="utf-8") as fh:  # type: ignore[arg-type]
        text = fh.read()
    assert "import calibration" not in text and "from .calibration" not in text
