"""Regenerate silly_kicks/tracking/preprocess/_provider_defaults_generated.py
from tests/fixtures/baselines/preprocess_baseline.json.

Run after every probe-baseline JSON change::

    uv run python scripts/regenerate_provider_defaults.py

The generated file is committed and consumed by silly_kicks.tracking.preprocess._config.
The integrity test (tests/test_preprocess_baseline_integrity.py) verifies that the
generated file matches the JSON within rel_tol=1e-6 -- i.e., regen happened.

Memory: feedback_codegen_for_data_to_code_integrity. PR-S24 lakehouse review S1 fix.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_JSON = REPO_ROOT / "tests" / "fixtures" / "baselines" / "preprocess_baseline.json"
OUTPUT_PY = REPO_ROOT / "silly_kicks" / "tracking" / "preprocess" / "_provider_defaults_generated.py"

_HEADER = '''"""AUTO-GENERATED -- DO NOT EDIT.

Source: tests/fixtures/baselines/preprocess_baseline.json
Regen:  uv run python scripts/regenerate_provider_defaults.py

PR-S24 lakehouse review S1 fix -- codegen pipeline replaces manual sync hand-edit.
"""

from __future__ import annotations

from ._config_dataclass import PreprocessConfig

_PROVIDER_DEFAULTS: dict[str, PreprocessConfig] = {
'''

_FOOTER = "}\n"


def _emit_block(provider: str, block: dict) -> str:
    derived = block["_derived_defaults"]
    return (
        f'    "{provider}": PreprocessConfig(\n'
        f'        smoothing_method="savgol",\n'
        f"        sg_window_seconds={derived['sg_window_seconds']},\n"
        f"        sg_poly_order={derived['sg_poly_order']},\n"
        f"        ema_alpha={derived['ema_alpha']},\n"
        f'        interpolation_method="linear",\n'
        f"        max_gap_seconds={derived['max_gap_seconds']},\n"
        f"        derive_velocity=True,\n"
        f"        link_quality_high_threshold={block['link_quality_high_threshold']},\n"
        f"    ),\n"
    )


def main() -> int:
    payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    OUTPUT_PY.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(_emit_block(p, payload[p]) for p in ("sportec", "pff", "metrica", "skillcorner"))
    OUTPUT_PY.write_text(_HEADER + body + _FOOTER, encoding="utf-8")
    print(f"[regen] wrote {OUTPUT_PY.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
